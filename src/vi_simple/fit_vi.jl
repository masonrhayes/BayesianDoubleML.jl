# Variational Inference fitting functions for BDML
# Standard full-batch VI for small to medium datasets
# Uses Turing's native variational inference with ADVI

using Turing.Variational
using ADTypes

export fit_bdml_vi_simple_legacy, extract_alpha_vi_simple, credible_interval

"""
    fit_bdml_vi(Y, D, X; kwargs...)

Fit BDML model using Variational Inference (ADVI) for fast approximate inference.

This is the standard full-batch VI implementation for small to medium datasets.
For large datasets (n > 10,000), consider using `fit_bdml_vi_subsampled()` instead.

# Arguments
- `Y::Vector{Float64}`: Outcome variable
- `D::Vector{Float64}`: Treatment variable
- `X::Matrix{Float64}`: Control variables (covariates)

# Keyword Arguments
- `model_type::Symbol`: :hier (hierarchical) or :basic (default: :hier)
- `n_vi_iterations::Int`: Number of VI optimization steps (default: 1000)
- `n_draws::Int`: Number of samples to draw from variational posterior (default: 2000)
- `ad_type`: AD backend type (default: AutoMooncake - fastest after warmup)
- `ad_kwargs::NamedTuple`: Keyword arguments for AD backend constructor (default: (;))
- `show_progress::Bool`: Show progress bar (default: true)

# AD Backend Options
- `AutoMooncake`: **Default and recommended**. Reverse-mode AD with rule compilation.
  - **⚠️ Requires warmup**: First few runs compile differntiation rules.
  - After warmup, typically 5-10x faster than ReverseDiff
  - No additional kwargs needed
- `AutoReverseDiff`: Reverse-mode AD with tape compilation
  - `ad_kwargs=(compile=true,)`: Compile tape for faster repeated evaluations
  - Good if you need consistent performance without warmup
- `AutoForwardDiff`: Forward-mode AD, good for small models
  - `ad_kwargs=(chunksize=0,)`: Chunk size for forward mode (0=automatic)
- `AutoZygote`: Alternative reverse-mode AD using Zygote
- `AutoEnzyme`: LLVM-based AD (currently unavailable on Julia 1.12+ due to compatibility issues)

# Returns
`BDMLVIResult`: Struct containing variational posterior, alpha samples, and convergence info

# Examples
```julia
# Default: AutoMooncake (fastest after warmup)
result = fit_bdml_vi(Y, D, X)

# For production/benchmarking - run warmup first
result_warmup = fit_bdml_vi(Y, D, X; n_vi_iterations=50)  # Compile rules
result = fit_bdml_vi(Y, D, X; n_vi_iterations=1000)       # Fast execution

# AutoReverseDiff without warmup needed
result = fit_bdml_vi(Y, D, X; ad_type=AutoReverseDiff, ad_kwargs=(compile=true,))

# AutoForwardDiff for small models
result = fit_bdml_vi(Y, D, X; ad_type=AutoForwardDiff)
```

# Performance Notes
- **AutoMooncake**: ~5-10x faster than ReverseDiff after warmup (0.08s vs 0.5s typical)
  - First run(s) compile custom differentiation rules for your model
  - Subsequent runs reuse compiled rules
  - Best for production use where you can afford initial warmup
- **AutoReverseDiff**: Consistent performance, good for one-off analyses
- Typical speedup over MCMC: 10-50x depending on problem size
- For large datasets (n > 10,000), use `fit_bdml_vi_subsampled()` instead

# Technical Details
- Uses VI-compatible models with Beta(2,2) correlation prior (instead of LKJCholesky)
- Correlation is ρ = 2*ρ_raw - 1 where ρ_raw ~ Beta(2, 2)
- VI samples are in constrained space (already transformed by Turing's bijectors)
- Uses Turing's native `vi()` function with ADVI algorithm

# See Also
- `fit()`: Generic fit function with dispatch
"""
function fit_bdml_vi_simple_legacy(
        Y, D, X;
        model_type::Symbol = :hier,
        n_vi_iterations::Int = 1000,
        n_draws::Int = 2000,
        ad_type = AutoMooncake,
        ad_kwargs::NamedTuple = (;),
        show_progress::Bool = true
    )

    @assert model_type in [:basic, :hier] "model_type must be :basic or :hier"

    # Standardize data
    Y_s, D_s, X_s, stats = standardize_data(Y, D, X)

    # Create VI-compatible model (uses Beta correlation, not LKJCholesky)
    model = if model_type == :hier
        bdml_hier_vi(Y_s, D_s, X_s)
    else
        bdml_basic_vi(Y_s, D_s, X_s)
    end

    # Construct AD backend with user-provided kwargs
    # Handle default compile=true for AutoReverseDiff
    if ad_type == AutoReverseDiff && !haskey(ad_kwargs, :compile)
        ad_kwargs = merge(ad_kwargs, (compile = true,))
    end

    # Warning for Mooncake about warmup requirements
    if ad_type == AutoMooncake
        @info "Using AutoMooncake AD backend. Note: First 1-2 runs compile differentiation rules. For optimal performance, run warmup iterations or check test_vi_ad_backends.jl for benchmarking guidance."
    end

    ad_backend = ad_type(; ad_kwargs...)

    # Run VI using Turing's variational inference with specified AD backend
    q_init = Variational.q_meanfield_gaussian(model)

    try
        q_result = vi(
            model, q_init, n_vi_iterations;
            show_progress = show_progress,
            adtype = ad_backend
        )

        # vi() returns (variational_distribution, stats) tuple
        q = q_result[1]
        stats_vi = q_result[2]

        # ELBO monitoring - extract from stats if available
        elbo_history = Float64[-1.0]  # Default placeholder
        if !isempty(stats_vi) && hasproperty(stats_vi[1], :elbo)
            elbo_history = [s.elbo for s in stats_vi]
        end
        final_elbo = length(elbo_history) > 0 ? elbo_history[end] : -1.0

        # Check ELBO convergence using dynamic window (last 30%, rel_tol=0.1%)
        converged, conv_msg = check_elbo_convergence(
            elbo_history;
            min_pct = 0.3,
            rel_tol = 0.001,
            check_trend = true,
            min_iterations = 50
        )

        @debug "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

        # Draw samples from variational posterior
        # Note: These samples are in CONSTRAINED space (already transformed by bijectors)
        vi_samples = rand(q, n_draws)

        # Extract alpha samples using verified parameter indexing
        α_s_samples = extract_alpha_vi_simple(vi_samples, size(X_s, 2), model_type)

        # Transform back to original scale
        scaling_factor = stats.Y_sd / stats.D_sd
        α_samples = α_s_samples .* scaling_factor

        return BDMLVIResult(
            q,
            α_samples,
            α_s_samples,
            stats,
            model_type,
            :meanfield,
            :simple,
            n_vi_iterations,
            elbo_history,
            converged,
            final_elbo
        )

    catch e
        @error "VI fitting failed with $(ad_type)" exception = (e, catch_backtrace())
        @info "You may try a different AD backend, e.g., fit_bdml_vi(Y, D, X; ad_type=AutoForwardDiff)"
        rethrow(e)
    end
end

"""
    fit_bdml_vi_simple(data::BDMLData; kwargs...)

Convenience method to fit VI using BDMLData struct.
"""
function fit_bdml_vi_simple_legacy(data::BDMLData; kwargs...)
    return fit_bdml_vi_simple_legacy(data.Y, data.D, data.X; kwargs...)
end

"""
    get_model_dimension(model)

Get the dimension (number of parameters) of a Turing model.
"""
function get_model_dimension(model)
    vi = DynamicPPL.VarInfo(model)
    return length(vi.values)
end

"""
    extract_alpha_vi_simple(vi_samples, p, model_type)

Extract α samples from VI posterior samples (simple implementation).

# Arguments
- `vi_samples::Matrix{Float64}`: Samples from variational posterior (in constrained space)
- `p::Int`: Number of control variables (covariates)
- `model_type::Symbol`: :hier or :basic

# Returns
- `Vector{Float64}`: Alpha samples (α = ρ * σ_U / σ_V)

# Notes
For VI models, the correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1].

The samples from `rand(q, n)` are in CONSTRAINED space, meaning:
- σ_U and σ_V are already positive (transformed by their bijectors)
- ρ_raw is already in [0, 1] (transformed by Beta bijector)

Parameter ordering in the flat array:
- Hierarchical: [σ²_δ, σ²_γ, θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
- Basic: [θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
"""
function extract_alpha_vi_simple(vi_samples, p::Int, model_type::Symbol)
    n_samples = size(vi_samples, 2)
    α_samples = Vector{Float64}(undef, n_samples)

    # Calculate parameter indices based on verified parameter ordering
    # Samples are in CONSTRAINED space (already bijected)

    if model_type == :hier
        # For hierarchical VI model: [σ²_δ, σ²_γ, θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw]
        # Total: 2 + 2*p + 2 + 1 = 2p + 5
        # Indices (1-indexed):
        #   σ²_δ = 1, σ²_γ = 2
        #   θ_Y = 3:3+p-1 = 3:2+p
        #   θ_D = 3+p:3+2p-1 = 3+p:2+2p
        #   σ_U = 3 + 2p
        #   σ_V = 4 + 2p
        #   ρ_raw = 5 + 2p
        σ_U_idx = 3 + 2 * p
        σ_V_idx = 4 + 2 * p
        ρ_raw_idx = 5 + 2 * p
    else
        # For basic VI model: [θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw]
        # Total: 2*p + 2 + 1 = 2p + 3
        # Indices (1-indexed):
        #   θ_Y = 1:p
        #   θ_D = 1+p:2p
        #   σ_U = 2p + 1
        #   σ_V = 2p + 2
        #   ρ_raw = 2p + 3
        σ_U_idx = 2 * p + 1
        σ_V_idx = 2 * p + 2
        ρ_raw_idx = 2 * p + 3
    end

    # Verify indices are within bounds
    d = size(vi_samples, 1)
    @assert σ_U_idx <= d "σ_U index ($σ_U_idx) exceeds dimension ($d)"
    @assert σ_V_idx <= d "σ_V index ($σ_V_idx) exceeds dimension ($d)"
    @assert ρ_raw_idx <= d "ρ_raw index ($ρ_raw_idx) exceeds dimension ($d)"

    for i in 1:n_samples
        # Extract parameters (already in constrained space)
        σ_U = vi_samples[σ_U_idx, i]
        σ_V = vi_samples[σ_V_idx, i]
        ρ_raw = vi_samples[ρ_raw_idx, i]

        # Transform ρ_raw from [0, 1] to [-1, 1]
        ρ = 2 * ρ_raw - 1

        # Compute alpha: α = ρ * σ_U / σ_V
        α_samples[i] = ρ * σ_U / σ_V
    end

    return α_samples
end

"""
    credible_interval(samples::Vector{Float64}; level=0.95)

Compute credible interval for a vector of samples.
"""
function credible_interval(samples::Vector{Float64}; level::Float64 = 0.95)
    α = 1 - level
    lower = α / 2
    upper = 1 - α / 2
    return quantile(samples, lower), quantile(samples, upper)
end

# Note: credible_interval(::BDMLVIResult) is defined in the unified implementation (src/vi/)
# We don't redefine it here to avoid method overwriting

# Extend existing credible_interval function for BDMLMCMCResult
function credible_interval(result::BDMLMCMCResult; level::Float64 = 0.95)
    return credible_interval(result.alpha_samples; level = level)
end
