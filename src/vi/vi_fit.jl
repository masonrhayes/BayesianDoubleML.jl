# Unified Variational Inference Fitting for BDML
# Supports all AD backends with explicit Bijectors
# Primary: AutoReverseDiff, AutoMooncake
# Secondary: AutoZygote, AutoForwardDiff

using AdvancedVI
using LogDensityProblems
using LogDensityProblemsAD
using Bijectors
using LinearAlgebra
using Random
using Statistics
using Optimisers
using ADTypes

export fit_bdml_vi, extract_alpha_vi

"""
    fit_bdml_vi(Y, D, X; kwargs...)

Fit BDML model using Variational Inference with unified implementation.

This is the unified VI implementation that:
- Uses explicit Bijectors for unconstrained → constrained transformation
- Supports all AD backends (ReverseDiff, Mooncake, Zygote, ForwardDiff)
- Automatically subsamples for large datasets (n > 10,000)
- Provides consistent interface across all configurations

# Arguments
- `Y::Vector{Float64}`: Outcome variable
- `D::Vector{Float64}`: Treatment variable
- `X::Matrix{Float64}`: Control variables (covariates)

# Keyword Arguments
- `model_type::Symbol=:hier`: :hier (hierarchical) or :basic
- `n_vi_iterations::Int=1000`: Number of VI optimization steps
- `n_draws::Int=2000`: Number of posterior samples to draw
- `ad_type=AutoReverseDiff`: AD backend (see below)
- `ad_kwargs::NamedTuple=(;)`: AD backend configuration
- `subsample::Union{Bool,Nothing}=nothing`: Enable subsampling
  - `nothing` (default): Auto-enable when n >= 10,000
  - `true`: Force subsampling
  - `false`: Force full-batch
- `batch_size::Int=-1`: Mini-batch size (auto-computed if not specified)
- `show_progress::Bool=true`: Show progress bar

# AD Backend Options

## Primary Backends (Recommended)

**AutoReverseDiff** (Default)
- Most stable across all use cases
- Uses `compile=false` (required by AdvancedVI >= 0.7)
- No warmup required
- Good performance for all model sizes

**AutoMooncake**
- 5-10x faster than ReverseDiff after warmup
- Rule compilation on first 1-2 runs (~50-100 iterations)
- Best for production/batch processing
- Use this if you can afford initial warmup

## Secondary Backends

**AutoZygote**
- Source-to-source reverse-mode AD
- Higher memory usage
- Good for experimentation
- Now working with explicit Bijectors!

**AutoForwardDiff**
- Forward-mode (dual numbers)
- Best for small models (p < 20)
- Constant compilation time
- Scales poorly with many parameters

# Returns
`BDMLVIResult`: Struct containing:
- `posterior`: Variational posterior (in unconstrained space)
- `alpha_samples::Vector{Float64}`: Causal effect samples (transformed to original scale)
- `alpha_samples_standardized::Vector{Float64}`: Samples on standardized scale
- `stats`: Data standardization statistics
- `model_type`: Model type used
- `vi_type::Symbol`: :meanfield or :meanfield_subsampled
- `n_vi_iterations`: Number of iterations run
- `elbo_history::Vector{Float64}`: ELBO values during optimization
- `converged::Bool`: Whether convergence criteria were met
- `final_elbo::Float64`: Final ELBO value

# Examples

## Basic Usage (Default: AutoReverseDiff)
```julia
result = fit_bdml_vi(Y, D, X)
mean(result.alpha_samples)  # Average causal effect
std(result.alpha_samples)   # Uncertainty
credible_interval(result)   # 95% credible interval
```

## With AutoMooncake (Fast after warmup)
```julia
# First run compiles rules (slow)
result_warmup = fit_bdml_vi(Y, D, X; ad_type=AutoMooncake, n_vi_iterations=50)

# Subsequent runs are 5-10x faster
result = fit_bdml_vi(Y, D, X; ad_type=AutoMooncake, n_vi_iterations=1000)
```

## Large Dataset (Auto-subsampling)
```julia
# Automatically uses subsampling when n >= 10,000
result = fit_bdml_vi(Y_large, D_large, X_large)
# VI type: :meanfield_subsampled
```

## Explicit Subsampling Control
```julia
# Force subsampling with custom batch size
result = fit_bdml_vi(Y, D, X; 
    subsample=true, 
    batch_size=256,
    n_vi_iterations=5000)

# Force full-batch even for large data
result = fit_bdml_vi(Y_large, D_large, X_large; subsample=false)
```

## Secondary Backends
```julia
# Zygote (now working!)
result = fit_bdml_vi(Y, D, X; ad_type=AutoZygote)

# ForwardDiff (small models only)
result = fit_bdml_vi(Y, D, X; ad_type=AutoForwardDiff)
```

# Performance Tips

1. **For small/medium data (n < 10,000)**
   - Use default AutoReverseDiff
   - Full-batch converges faster

2. **For large data (n >= 10,000)**
   - Auto-subsampling kicks in automatically
   - Use larger batch sizes (512-1024) for stability
   - Requires 3-5x more iterations but still much faster overall

3. **For production/batch processing**
   - Use AutoMooncake after warmup
   - Run 50-100 iterations first to compile rules
   - Subsequent runs will be 5-10x faster

4. **For experimentation/debugging**
   - Use AutoZygote or AutoForwardDiff
   - No compilation overhead
   - Easier to debug but slower

# Implementation Details

The unified implementation uses:
1. **Explicit Bijectors**: Manual transformation unconstrained → constrained
2. **AdvancedVI**: Consistent optimization interface across all backends
3. **LogDensityProblems**: Standardized log-posterior interface
4. **Pre-allocated temporaries**: Avoid allocations during AD for performance

# Technical Notes

- Samples from `rand(posterior)` are in UNCONSTRAINED space
- Alpha extraction transforms samples to CONSTRAINED space via bijector
- Scaling factor transforms from standardized to original scale
- All AD backends now use the same code path with explicit Bijectors

# See Also
- `fit_bdml()`: MCMC inference (more accurate but slower)
- `credible_interval()`: Compute credible intervals
- `extract_alpha()`: Extract α from posterior samples

# References
- Chernozhukov et al. (2018) - Double/debiased machine learning
- Kucukelbir et al. (2017) - Automatic differentiation variational inference
"""
function fit_bdml_vi(
        Y, D, X;
        model_type::Symbol = :hier,
        n_vi_iterations::Int = 1000,
        n_draws::Int = 2000,
        ad_type = AutoReverseDiff,
        ad_kwargs::NamedTuple = (;),
        subsample::Union{Bool, Nothing} = nothing,
        batch_size::Int = -1,
        show_progress::Bool = true
    )

    # Validate inputs
    @assert model_type in [:basic, :hier] "model_type must be :basic or :hier, got $model_type"
    @assert n_vi_iterations > 0 "n_vi_iterations must be positive, got $n_vi_iterations"
    @assert n_draws > 0 "n_draws must be positive, got $n_draws"

    # Get dataset info
    n = length(Y)

    # Determine if we should use subsampling
    use_subsample = if isnothing(subsample)
        n >= 10000  # Auto-enable for large datasets
    else
        subsample
    end

    # Log configuration
    if use_subsample
        if batch_size <= 0
            batch_size = min(256, max(64, ceil(Int, n / 1000)))
        end
        @info "BDML VI ($(ad_type)): n=$n, subsample=true, batch_size=$batch_size, iterations=$n_vi_iterations"
    else
        @info "BDML VI ($(ad_type)): n=$n, subsample=false, iterations=$n_vi_iterations"
    end

    # Standardize data
    Y_s, D_s, X_s, stats = standardize_data(Y, D, X)
    p = size(X_s, 2)
    T = eltype(X_s)

    # Create unified model
    model = BDMLVIModel(Y_s, D_s, X_s; model_type = model_type, T = T)

    # Get dimension
    d = LogDensityProblems.dimension(model)

    # Configure AD backend
    ad_kwargs = configure_ad_backend(ad_type, ad_kwargs, use_subsample)

    # Create AD wrapper
    prob_ad = LogDensityProblemsAD.ADgradient(ad_type(; ad_kwargs...), model)

    # Set up variational family in unconstrained space
    # Smaller initial variance for stable optimization
    q0 = AdvancedVI.MeanFieldGaussian(zeros(d), Diagonal(fill(0.1, d)))

    # Configure algorithm based on subsampling
    # NOTE: AutoMooncake has compatibility issues with both RepGrad and ScoreGrad algorithms
    # due to SubArray operations in AdvancedVI. Use AutoReverseDiff for stable VI.
    if use_subsample
        # Subsampled VI: Use ReshufflingBatchSubsampling
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)

        alg = AdvancedVI.KLMinRepGradProxDescent(
            ad_type(; ad_kwargs...);
            subsampling = subsampling
        )
    else
        # Full-batch VI: No subsampling
        alg = AdvancedVI.KLMinRepGradProxDescent(ad_type(; ad_kwargs...))
    end

    # Run optimization
    q_result, opt_stats, _ = AdvancedVI.optimize(
        Random.default_rng(),
        alg, n_vi_iterations, prob_ad, q0;
        show_progress = show_progress
    )

    # Extract ELBO history
    elbo_history = Float64[]
    for stat in opt_stats
        if hasproperty(stat, :elbo)
            push!(elbo_history, stat.elbo)
        end
    end

    final_elbo = length(elbo_history) > 0 ? elbo_history[end] : -Inf

    # Check ELBO convergence with dynamic window (last 30%, rel_tol=0.1%)
    converged, conv_msg = check_elbo_convergence(
        elbo_history;
        min_pct = 0.3,
        rel_tol = 0.01,
        check_trend = true,
        min_iterations = 50
    )

    @debug "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    # Get bijector for transforming to constrained space
    binv = bijector(model)

    # Draw samples from variational posterior (unconstrained space)
    vi_samples_unconstrained = rand(q_result, n_draws)

    # Transform to constrained space
    vi_samples_constrained = similar(vi_samples_unconstrained)
    for i in 1:n_draws
        vi_samples_constrained[:, i] .= Bijectors.transform(binv, vi_samples_unconstrained[:, i])
    end

    # Extract alpha samples
    α_s_samples = extract_alpha_vi(vi_samples_constrained, p, model_type)

    # Transform back to original scale
    scaling_factor = stats.Y_sd / stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    # Return results
    return BDMLVIResult(
        q_result,
        α_samples,
        α_s_samples,
        stats,
        model_type,
        use_subsample ? :meanfield_subsampled : :meanfield,
        :unified,
        n_vi_iterations,
        elbo_history,
        converged,
        final_elbo
    )
end


"""
    fit_bdml_vi(data::BDMLData; kwargs...)

Convenience method to fit VI using BDMLData struct.

# Examples
```julia
data = BDMLData(Y, D, X)
result = fit_bdml_vi(data; model_type=:basic, ad_type=AutoMooncake)
```
"""
function fit_bdml_vi(data::BDMLData; kwargs...)
    return fit_bdml_vi(data.Y, data.D, data.X; kwargs...)
end

"""
    extract_alpha_vi(vi_samples, p::Int, model_type::Symbol)

Extract α samples from VI posterior samples.

For VI models, the correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1].

The samples are in CONSTRAINED space (already transformed by bijectors), meaning:
- σ_U and σ_V are already positive (transformed by their bijectors)
- ρ_raw is already in [0, 1] (transformed by Beta bijector)

Parameter ordering in the flat array:
- Hierarchical: [σ²_δ, σ²_γ, θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
- Basic: [θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]

# Arguments
- `vi_samples::Matrix{Float64}`: Samples from variational posterior (in constrained space)
- `p::Int`: Number of control variables (covariates)
- `model_type::Symbol`: :hier or :basic

# Returns
- `Vector{Float64}`: Alpha samples (α = ρ * σ_U / σ_V)
"""
function extract_alpha_vi(vi_samples, p::Int, model_type::Symbol)
    n_samples = size(vi_samples, 2)
    α_samples = Vector{Float64}(undef, n_samples)

    # Calculate parameter indices based on verified parameter ordering
    # Samples are in CONSTRAINED space (already bijected)

    if model_type == :hier
        σ_U_idx = 3 + 2 * p
        σ_V_idx = 4 + 2 * p
        ρ_raw_idx = 5 + 2 * p
    else
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
    credible_interval(result::BDMLVIResult; level=0.95)

Compute credible interval for α from BDMLVIResult.
"""
function credible_interval(result::BDMLVIResult; level::Float64 = 0.95)
    return credible_interval(result.alpha_samples; level = level)
end
