# Multiple Dispatch-Based Fitting Interface for BDML
#
# This file provides a unified `fit!()` interface that mutates the model
# and stores results internally. The model serves as both specification
# and result container.
#
# The actual implementation dispatches on both:
# 1. Model type (BDMLBasicModel vs BDMLHierarchicalModel)
# 2. Inference method (MCMCMethod, UnifiedVIMethod, SimpleVIMethod, or VMPMethod)

export fit!

"""
    fit!(model::AbstractBDMLModel, method::AbstractInferenceMethod; force=false, kwargs...)

Fit a BDML model using the specified inference method, storing results in the model.

This is a mutating function that modifies the model in-place. After fitting,
results can be extracted using `coeftable()`, `extract_alpha()`, `summary()`, etc.

If the model has already been fitted, a warning is shown unless `force=true`
is passed to allow refitting.

# Arguments
- `model::AbstractBDMLModel`: The model to fit (data + metadata)
- `method::AbstractInferenceMethod`: The inference algorithm to use

# Keyword Arguments
- `force::Bool=false`: Allow refitting an already-fitted model

For MCMC methods:
- `n_samples::Int=2000`: Number of posterior samples to draw
- `n_chains::Int=4`: Number of MCMC chains to run

For VI methods:
- `n_iterations::Int=1000`: Number of optimization iterations
- `n_draws::Int=2000`: Number of posterior samples to draw after fitting

# Returns
`nothing` (follows standard Julia mutating function convention)

# Examples
```julia
# Create model
model = BDMLModel(Y, D, X; model_type=:basic)

# Fit with MCMC (NUTS)
fit!(model, MCMCMethod(:nuts); n_samples=2000, n_chains=4)

# Fit with VI (Unified)
fit!(model, UnifiedVIMethod(); n_iterations=1000)

# Extract results
coef_table = coeftable(model)
summary(model)

# Refit with force (changes the stored result)
fit!(model, SimpleVIMethod(); force=true)
```

# Multiple Dispatch
The actual implementation dispatches on both model and method:
- `_fit_impl(::BDMLBasicModel, ::MCMCMethod)` - Basic model MCMC
- `_fit_impl(::BDMLHierarchicalModel, ::MCMCMethod)` - Hierarchical MCMC  
- `_fit_impl(::BDMLBasicModel, ::UnifiedVIMethod)` - Basic model VI (Unified)
- `_fit_impl(::BDMLHierarchicalModel, ::UnifiedVIMethod)` - Hierarchical VI (Unified)
- `_fit_impl(::BDMLBasicModel, ::SimpleVIMethod)` - Basic model VI (Simple)
- `_fit_impl(::BDMLHierarchicalModel, ::SimpleVIMethod)` - Hierarchical VI (Simple)
- `_fit_impl(::BDMLBasicModel, ::VMPMethod)` - Basic conjugate VMP (RxInfer extension)
- `_fit_impl(::BDMLHierarchicalModel, ::VMPMethod)` - Hierarchical conjugate VMP (RxInfer extension)

See also: [`BDMLModel`](@ref), [`MCMCMethod`](@ref), [`UnifiedVIMethod`](@ref), [`isfitted`](@ref)
"""
function fit!(
        model::AbstractBDMLModel, method::AbstractInferenceMethod;
        force::Bool = false, kwargs...
    )
    # Check if already fitted
    if model.is_fitted && !force
        @warn "Model has already been fitted. Use force=true to refit."
        return nothing
    end

    # Dispatch to implementation based on model type and method
    result = _fit_impl(model, method; kwargs...)

    # Store result in model
    model.result = result
    model.is_fitted = true
    model.last_method = method

    return nothing
end

"""
    fit!(model::AbstractBDMLModel; force=false, kwargs...)

Fit a model using default method (MCMC with NUTS).

Convenience method that defaults to NUTS sampler.
"""
function fit!(model::AbstractBDMLModel; force::Bool = false, kwargs...)
    return fit!(model, MCMCMethod(:nuts); force = force, kwargs...)
end

# Error fallback for unimplemented combinations
function _fit_impl(model::AbstractBDMLModel, method::AbstractInferenceMethod; kwargs...)
    error("No fit implementation defined for model type $(typeof(model)) with method $(typeof(method))")
end

# VMP routing via backend dispatch
function _fit_impl(model::AbstractBDMLModel, method::VMPMethod; kwargs...)
    return _fit_vmp(model, method; kwargs...)
end

# Fallback when a backend extension is not loaded
function _fit_vmp(model::AbstractBDMLModel, method::VMPMethod; kwargs...)
    return error(
        "VMP backend $(typeof(method.backend)) is not available. " *
        "Load the required extension (e.g., `using RxInfer`) and try again."
    )
end

# Variational family initialization

"""
    initialize_variational_distribution(dim::Int, family::AbstractVariationalFamily)

Initialize the variational distribution q0 based on the family type using multiple dispatch.
"""
function initialize_variational_distribution end

"""
    initialize_variational_distribution(dim::Int, ::MeanField)

Initialize a MeanField Gaussian with diagonal covariance.
"""
function initialize_variational_distribution(dim::Int, ::MeanField)
    return AdvancedVI.MeanFieldGaussian(zeros(dim), Diagonal(fill(0.1, dim)))
end

"""
    initialize_variational_distribution(dim::Int, family::LowRank)

Initialize a LowRank Gaussian with specified rank.

Uses diagonal + low-rank decomposition: Σ = D² + U*U'
Uses random initialization for low-rank factors to break symmetry.
"""
function initialize_variational_distribution(dim::Int, family::LowRank)
    actual_rank = min(family.rank, dim - 1)  # Cap rank at dim-1
    D = fill(0.1, dim)                    # Diagonal scale
    U = randn(dim, actual_rank) * 0.01    # Low-rank factors (random to break symmetry)
    return AdvancedVI.LowRankGaussian(zeros(dim), D, U)
end

"""
    initialize_variational_distribution(dim::Int, family::LowRankScore)

Initialize a LowRank Gaussian with specified rank for score gradient estimator.

Uses diagonal + low-rank decomposition: Σ = D² + U*U'
Uses random initialization for low-rank factors to break symmetry.
"""
function initialize_variational_distribution(dim::Int, family::LowRankScore)
    actual_rank = min(family.rank, dim - 1)  # Cap rank at dim-1
    D = fill(0.1, dim)                    # Diagonal scale
    U = randn(dim, actual_rank) * 0.01    # Low-rank factors (random to break symmetry)
    return AdvancedVI.LowRankGaussian(zeros(dim), D, U)
end

"""
    family_symbol(family::AbstractVariationalFamily)

Get the symbol representation of the variational family for result storage.
"""
family_symbol(::MeanField) = :meanfield
family_symbol(::LowRank) = :lowrank
family_symbol(::LowRankScore) = :lowrank_score

"""
    configure_vi_algorithm(method::UnifiedVIMethod, use_subsample::Bool, batch_size::Int, n::Int)

Configure the AdvancedVI algorithm based on the variational family using dispatch.

Different families require different algorithms:
- MeanField: Uses KLMinRepGradProxDescent with ProximalLocationScaleEntropy
- LowRank: Uses KLMinRepGradDescent with ClipScale operator
"""
function configure_vi_algorithm end

"""
    configure_vi_algorithm(method::UnifiedVIMethod{MeanField}, use_subsample::Bool, batch_size::Int, n::Int)

Configure KLMinRepGradProxDescent for MeanField families.
"""
function configure_vi_algorithm(method::UnifiedVIMethod{MeanField}, use_subsample::Bool, batch_size::Int, n::Int)
    ad_kwargs = ad_backend_kwargs(method.ad_backend)

    if use_subsample
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)
        return AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            subsampling = subsampling,
            n_samples = method.n_montecarlo
        )
    else
        return AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            n_samples = method.n_montecarlo
        )
    end
end

"""
    configure_vi_algorithm(method::UnifiedVIMethod{LowRank}, use_subsample::Bool, batch_size::Int, n::Int)

Configure KLMinRepGradDescent with Adam optimizer for LowRank families.

Uses Adam optimizer which can work better with low-rank structure than DoWG.
"""
function configure_vi_algorithm(method::UnifiedVIMethod{LowRank}, use_subsample::Bool, batch_size::Int, n::Int)
    ad_kwargs = ad_backend_kwargs(method.ad_backend)

    if use_subsample
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)
        return AdvancedVI.KLMinRepGradDescent(
            method.ad_backend(; ad_kwargs...);
            subsampling = subsampling,
            n_samples = method.n_montecarlo,
            optimizer = Optimisers.Adam(0.01),  # Use Adam instead of DoWG
            operator = AdvancedVI.ClipScale()  # ClipScale supports LowRank
        )
    else
        return AdvancedVI.KLMinRepGradDescent(
            method.ad_backend(; ad_kwargs...);
            n_samples = method.n_montecarlo,
            optimizer = Optimisers.Adam(0.01),  # Use Adam instead of DoWG
            operator = AdvancedVI.ClipScale()  # ClipScale supports LowRank
        )
    end
end

"""
    configure_vi_algorithm(method::UnifiedVIMethod{LowRankScore}, use_subsample::Bool, batch_size::Int, n::Int)

Configure KLMinScoreGradDescent (BBVI) for LowRankScore families.

Uses score gradient (REINFORCE) estimator with VarGrad control variate.
This can be more stable than reparameterization gradient for some problems.
"""
function configure_vi_algorithm(method::UnifiedVIMethod{LowRankScore}, use_subsample::Bool, batch_size::Int, n::Int)
    ad_kwargs = ad_backend_kwargs(method.ad_backend)

    if use_subsample
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)
        return AdvancedVI.KLMinScoreGradDescent(
            method.ad_backend(; ad_kwargs...);
            subsampling = subsampling,
            n_samples = method.n_montecarlo,
            operator = AdvancedVI.ClipScale()  # ClipScale supports LowRank
        )
    else
        return AdvancedVI.KLMinScoreGradDescent(
            method.ad_backend(; ad_kwargs...);
            n_samples = method.n_montecarlo,
            operator = AdvancedVI.ClipScale()  # ClipScale supports LowRank
        )
    end
end

# MCMC dispatch

"""
    _fit_impl(model::BDMLBasicModel, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit basic BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization (Turing's native approach).
"""
function _fit_impl(
        model::BDMLBasicModel, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )

    # Create Turing model (uses LKJCholesky for MCMC)
    turing_model = bdml_basic(model.Y, model.D, model.X)

    # Create sampler based on method
    if method.algorithm == :nuts
        # Use Turing's NUTS with parameters from method
        mcmc_sampler = NUTS(method.target_acceptance; adtype = AutoForwardDiff())
    else
        error("Unknown MCMC algorithm: $(method.algorithm). Only :nuts is supported.")
    end

    # Run MCMC
    chain = if n_chains == 1
        sample(turing_model, mcmc_sampler, n_samples; progress = true)
    else
        sample(turing_model, mcmc_sampler, MCMCThreads(), n_samples, n_chains; progress = true)
    end

    # Extract alpha
    α_s_samples = extract_alpha(chain)

    # Transform back to original scale
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLMCMCResult(chain, α_samples, α_s_samples, model.stats, :basic)
end

"""
    _fit_impl(model::BDMLHierarchicalModel, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit hierarchical BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization.
"""
function _fit_impl(
        model::BDMLHierarchicalModel, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )

    # Create Turing model
    turing_model = bdml_hier(model.Y, model.D, model.X)

    # Create sampler
    if method.algorithm == :nuts
        mcmc_sampler = NUTS(method.target_acceptance; adtype = AutoForwardDiff())
    else
        error("HMC via dispatch interface not yet implemented.")
    end

    # Run MCMC
    chain = if n_chains == 1
        sample(turing_model, mcmc_sampler, n_samples; progress = true)
    else
        sample(turing_model, mcmc_sampler, MCMCThreads(), n_samples, n_chains; progress = true)
    end

    # Extract alpha
    α_s_samples = extract_alpha(chain)

    # Transform back
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLMCMCResult(chain, α_samples, α_s_samples, model.stats, :hier)
end

# VI dispatch - Unified

"""
    _fit_impl(model::BDMLBasicModel, method::UnifiedVIMethod; n_iterations=1000, n_draws=2000)

Fit basic BDML model using Unified VI (AdvancedVI with explicit Bijectors).

Uses Beta(2,2) correlation parameterization (VI-compatible).
Supports subsampling for large datasets.
"""
function _fit_impl(
        model::BDMLBasicModel, method::UnifiedVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create unified VI model (uses explicit bijectors)
    vi_model = BDMLVIModel(model.Y, model.D, model.X; model_type = :basic, T = Float64)

    # Determine subsampling
    n = nobs(model)
    use_subsample = if isnothing(method.subsample)
        n >= 10000  # Auto-enable
    else
        method.subsample
    end

    # Auto-compute batch size if needed
    batch_size = if use_subsample && method.batch_size <= 0
        min(256, max(64, ceil(Int, n / 1000)))
    else
        method.batch_size
    end

    # Log configuration
    if use_subsample
        @info "BDML VI ($(method.ad_backend)): n=$n, subsample=true, batch_size=$batch_size, iterations=$n_iterations"
    else
        @info "BDML VI ($(method.ad_backend)): n=$n, subsample=false, iterations=$n_iterations"
    end

    # Get dimension
    d = LogDensityProblems.dimension(vi_model)
    p = ncovariates(model)

    # Configure AD backend (compile=false for ReverseDiff, warmup notice for Mooncake)
    ad_kwargs = configure_ad_backend(method.ad_backend, (;), use_subsample)

    # Create AD wrapper
    prob_ad = LogDensityProblemsAD.ADgradient(method.ad_backend(; ad_kwargs...), vi_model)

    # Set up variational family using dispatch based on family type
    q0 = initialize_variational_distribution(d, method.family)

    # Configure algorithm using dispatch (different algorithms for different families)
    alg = configure_vi_algorithm(method, use_subsample, batch_size, n)

    # Run optimization
    q_result, opt_stats, _ = AdvancedVI.optimize(
        Random.default_rng(),
        alg, n_iterations, prob_ad, q0;
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

    # Check ELBO convergence with dynamic window (last 30%, rel_tol=5%)
    converged, conv_msg = check_elbo_convergence(
        elbo_history;
        min_pct = 0.3,
        rel_tol = 0.05,
        check_trend = true,
        min_iterations = 50
    )

    @info "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    # Get bijector and transform samples
    binv = bijector(vi_model)
    vi_samples_unconstrained = rand(q_result, n_draws)
    vi_samples_constrained = similar(vi_samples_unconstrained)
    for i in 1:n_draws
        vi_samples_constrained[:, i] .= Bijectors.transform(binv, vi_samples_unconstrained[:, i])
    end

    # Extract alpha
    α_s_samples = extract_alpha(vi_samples_constrained, p, :basic)

    # Transform back
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    # Determine result type based on family and subsampling
    base_vi_type = family_symbol(method.family)
    vi_type = use_subsample ? Symbol(base_vi_type, :_subsampled) : base_vi_type

    return BDMLVIResult(
        q_result, α_samples, α_s_samples, model.stats,
        :basic, vi_type, :unified, n_iterations, elbo_history, converged, final_elbo
    )
end

"""
    _fit_impl(model::BDMLHierarchicalModel, method::UnifiedVIMethod; n_iterations=1000, n_draws=2000)

Fit hierarchical BDML model using Unified VI.
"""
function _fit_impl(
        model::BDMLHierarchicalModel, method::UnifiedVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create unified VI model
    vi_model = BDMLVIModel(model.Y, model.D, model.X; model_type = :hier, T = Float64)

    # Determine subsampling
    n = nobs(model)
    use_subsample = if isnothing(method.subsample)
        n >= 10000
    else
        method.subsample
    end

    batch_size = if use_subsample && method.batch_size <= 0
        min(256, max(64, ceil(Int, n / 1000)))
    else
        method.batch_size
    end

    if use_subsample
        @info "BDML VI ($(method.ad_backend)): n=$n, subsample=true, batch_size=$batch_size, iterations=$n_iterations"
    else
        @info "BDML VI ($(method.ad_backend)): n=$n, subsample=false, iterations=$n_iterations"
    end

    d = LogDensityProblems.dimension(vi_model)
    p = ncovariates(model)

    # Configure AD backend (compile=false for ReverseDiff, warmup notice for Mooncake)
    ad_kwargs = configure_ad_backend(method.ad_backend, (;), use_subsample)

    prob_ad = LogDensityProblemsAD.ADgradient(method.ad_backend(; ad_kwargs...), vi_model)

    # Set up variational family using dispatch based on family type
    q0 = initialize_variational_distribution(d, method.family)

    # Configure algorithm using dispatch (different algorithms for different families)
    alg = configure_vi_algorithm(method, use_subsample, batch_size, n)

    q_result, opt_stats, _ = AdvancedVI.optimize(
        Random.default_rng(),
        alg, n_iterations, prob_ad, q0;
        show_progress = show_progress
    )

    elbo_history = Float64[]
    for stat in opt_stats
        if hasproperty(stat, :elbo)
            push!(elbo_history, stat.elbo)
        end
    end
    final_elbo = length(elbo_history) > 0 ? elbo_history[end] : -Inf

    # Check ELBO convergence with dynamic window (last 30%, rel_tol=5%)
    converged, conv_msg = check_elbo_convergence(
        elbo_history;
        min_pct = 0.3,
        rel_tol = 0.05,
        check_trend = true,
        min_iterations = 50
    )

    @info "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    binv = bijector(vi_model)
    vi_samples_unconstrained = rand(q_result, n_draws)
    vi_samples_constrained = similar(vi_samples_unconstrained)
    for i in 1:n_draws
        vi_samples_constrained[:, i] .= Bijectors.transform(binv, vi_samples_unconstrained[:, i])
    end

    # Parameter indices for hierarchical
    σ_U_idx = 3 + 2 * p
    σ_V_idx = 4 + 2 * p
    ρ_raw_idx = 5 + 2 * p

    α_s_samples = Vector{Float64}(undef, n_draws)
    for i in 1:n_draws
        σ_U = vi_samples_constrained[σ_U_idx, i]
        σ_V = vi_samples_constrained[σ_V_idx, i]
        ρ_raw = vi_samples_constrained[ρ_raw_idx, i]
        ρ = 2 * ρ_raw - 1
        α_s_samples[i] = ρ * σ_U / σ_V
    end

    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    # Determine result type based on family and subsampling
    base_vi_type = family_symbol(method.family)
    vi_type = use_subsample ? Symbol(base_vi_type, :_subsampled) : base_vi_type

    return BDMLVIResult(
        q_result, α_samples, α_s_samples, model.stats,
        :hier, vi_type, :unified, n_iterations, elbo_history, converged, final_elbo
    )
end

# VI dispatch - Simple

"""
    _fit_impl(model::BDMLBasicModel, method::SimpleVIMethod; n_iterations=1000, n_draws=2000)

Fit basic BDML model using Simple VI (Turing's native vi() function).

This works well with AutoMooncake. No subsampling support.
"""
function _fit_impl(model::BDMLBasicModel, method::SimpleVIMethod; kwargs...)
    return _fit_simple_vi(bdml_basic_vi, model, method, :basic; kwargs...)
end

"""
    _fit_impl(model::BDMLHierarchicalModel, method::SimpleVIMethod; n_iterations=1000, n_draws=2000)

Fit hierarchical BDML model using Simple VI.
"""
function _fit_impl(model::BDMLHierarchicalModel, method::SimpleVIMethod; kwargs...)
    return _fit_simple_vi(bdml_hier_vi, model, method, :hier; kwargs...)
end

"""
    _fit_simple_vi(model_constructor, model, method, model_type; n_iterations=1000, n_draws=2000)

Shared implementation for Simple VI fits via Turing's native `vi()` with a
mean-field Gaussian variational family.
"""
function _fit_simple_vi(
        model_constructor, model::AbstractBDMLModel, method::SimpleVIMethod, model_type::Symbol;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create Turing model (uses Beta correlation for VI compatibility)
    turing_model = model_constructor(model.Y, model.D, model.X)

    # Configure AD backend (compile=false for ReverseDiff, warmup notice for Mooncake)
    ad_kwargs = configure_ad_backend(method.ad_backend, (;), false)

    # Run VI using Turing 0.46+ API: pass q_meanfield_gaussian as a function
    q_result = vi(
        turing_model, Variational.q_meanfield_gaussian, n_iterations;
        show_progress = show_progress,
        adtype = method.ad_backend(; ad_kwargs...)
    )

    q = q_result.q
    stats_vi = q_result.info

    # Extract ELBO history
    elbo_history = Float64[-1.0]
    if !isempty(stats_vi) && hasproperty(stats_vi[1], :elbo)
        elbo_history = [s.elbo for s in stats_vi]
    end
    final_elbo = length(elbo_history) > 0 ? elbo_history[end] : -1.0

    # Check ELBO convergence with dynamic window (last 30%, rel_tol=5%)
    converged, conv_msg = check_elbo_convergence(
        elbo_history;
        min_pct = 0.3,
        rel_tol = 0.05,
        check_trend = true,
        min_iterations = 50
    )

    @info "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    # Draw constrained samples as VarNamedTuples (Turing 0.46+)
    vnt_samples = rand(q_result, n_draws)

    # Extract alpha from constrained VNT samples: α = (2ρ_raw - 1) * σ_U / σ_V
    α_s_samples = extract_alpha(vnt_samples)

    # Transform back
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLVIResult(
        q, α_samples, α_s_samples, model.stats,
        model_type, :meanfield, :simple, n_iterations, elbo_history, converged, final_elbo
    )
end
