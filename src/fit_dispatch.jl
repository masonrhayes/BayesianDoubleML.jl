# Multiple Dispatch-Based Fitting Interface for BDML
#
# This file provides a unified `fit()` interface that dispatches on both:
# 1. Problem type (BDMLBasicProblem vs BDMLHierarchicalProblem)
# 2. Inference method (MCMCMethod vs UnifiedVIMethod vs SimpleVIMethod)
#
# This leverages Julia's multiple dispatch to have clean, extensible code
# where each combination has its own implementation logic.

export fit

# Import the old fitting functions - we'll wrap them in dispatch
# These are defined in fit.jl, vi/vi_fit.jl, and vi_simple/fit_vi.jl

"""
    fit(problem::AbstractBDMLProblem, method::AbstractInferenceMethod; kwargs...)

Fit a BDML problem using the specified inference method.

This is the core multiple dispatch function that routes to appropriate
algorithm implementations based on both problem type and method type.

# Arguments
- `problem::AbstractBDMLProblem`: The problem specification (data + metadata)
- `method::AbstractInferenceMethod`: The inference algorithm to use

# Keyword Arguments (method-dependent)
For MCMC:
- `n_samples::Int=2000`: Number of posterior samples to draw
- `n_chains::Int=4`: Number of MCMC chains to run

For VI:
- `n_iterations::Int=1000`: Number of optimization iterations
- `n_draws::Int=2000`: Number of posterior samples to draw after fitting

# Returns
`BDMLMCMCResult` or `BDMLVIResult` depending on method type.

# Examples
```julia
# Create problem
prob = BDMLProblem(Y, D, X; model_type=:basic)

# Fit with MCMC (NUTS)
result = fit(prob, MCMCMethod(:nuts); n_samples=2000, n_chains=4)

# Fit with VI (Unified)
result = fit(prob, UnifiedVIMethod(); n_iterations=1000)

# Fit with VI (Simple)
result = fit(prob, SimpleVIMethod(); n_iterations=1000)

# Default fit uses MCMC with NUTS
result = fit(prob)  # Same as fit(prob, MCMCMethod(:nuts))
```

# Multiple Dispatch
The actual implementation is dispatched based on both arguments:
- `fit(::BDMLBasicProblem, ::MCMCMethod)` - Basic model MCMC
- `fit(::BDMLHierarchicalProblem, ::MCMCMethod)` - Hierarchical MCMC  
- `fit(::BDMLBasicProblem, ::UnifiedVIMethod)` - Basic model VI (Unified)
- `fit(::BDMLHierarchicalProblem, ::UnifiedVIMethod)` - Hierarchical VI (Unified)
- `fit(::BDMLBasicProblem, ::SimpleVIMethod)` - Basic model VI (Simple)
- `fit(::BDMLHierarchicalProblem, ::SimpleVIMethod)` - Hierarchical VI (Simple)

See also: [`BDMLProblem`](@ref), [`MCMCMethod`](@ref), [`UnifiedVIMethod`](@ref)
"""
function fit(problem::AbstractBDMLProblem, method::AbstractInferenceMethod; kwargs...)
    error("No fit method defined for problem type $(typeof(problem)) with method $(typeof(method))")
end

"""
    fit(problem::AbstractBDMLProblem; kwargs...)

Fit a problem using default method (MCMC with NUTS).

Convenience method that defaults to NUTS sampler.
"""
fit(problem::AbstractBDMLProblem; kwargs...) = fit(problem, MCMCMethod(:nuts); kwargs...)

# ==================== MCMC DISPATCH ====================

"""
    fit(prob::BDMLBasicProblem, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit basic BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization (Turing's native approach).
"""
function fit(
        prob::BDMLBasicProblem, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )

    # Create Turing model (uses LKJCholesky for MCMC)
    model = bdml_basic(prob.Y, prob.D, prob.X)

    # Create sampler based on method
    if method.algorithm == :nuts
        # Use Turing's NUTS with parameters from method
        mcmc_sampler = NUTS(method.target_acceptance; adtype = AutoForwardDiff())
    elseif method.algorithm == :hmc
        # HMC not yet implemented via this interface
        error("HMC via dispatch interface not yet implemented. Use fit_bdml() directly.")
    else
        error("Unknown MCMC algorithm: $(method.algorithm)")
    end

    # Run MCMC
    chain = if n_chains == 1
        sample(model, mcmc_sampler, n_samples; progress = true)
    else
        sample(model, mcmc_sampler, MCMCThreads(), n_samples, n_chains; progress = true)
    end

    # Extract alpha
    α_s_samples = extract_alpha(chain)

    # Transform back to original scale
    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLMCMCResult(chain, α_samples, α_s_samples, prob.stats, :basic)
end

"""
    fit(prob::BDMLHierarchicalProblem, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit hierarchical BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization.
"""
function fit(
        prob::BDMLHierarchicalProblem, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )

    # Create Turing model
    model = bdml_hier(prob.Y, prob.D, prob.X)

    # Create sampler
    if method.algorithm == :nuts
        mcmc_sampler = NUTS(method.target_acceptance; adtype = AutoForwardDiff())
    else
        error("HMC via dispatch interface not yet implemented.")
    end

    # Run MCMC
    chain = if n_chains == 1
        sample(model, mcmc_sampler, n_samples; progress = true)
    else
        sample(model, mcmc_sampler, MCMCThreads(), n_samples, n_chains; progress = true)
    end

    # Extract alpha
    α_s_samples = extract_alpha(chain)

    # Transform back
    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLMCMCResult(chain, α_samples, α_s_samples, prob.stats, :hier)
end

# ==================== VI DISPATCH - UNIFIED ====================

"""
    fit(prob::BDMLBasicProblem, method::UnifiedVIMethod; n_iterations=1000, n_draws=2000)

Fit basic BDML model using Unified VI (AdvancedVI with explicit Bijectors).

Uses Beta(2,2) correlation parameterization (VI-compatible).
Supports subsampling for large datasets.
"""
function fit(
        prob::BDMLBasicProblem, method::UnifiedVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create unified VI model (uses explicit bijectors)
    vi_model = BDMLVIModel(prob.Y, prob.D, prob.X; model_type = :basic, T = Float64)

    # Determine subsampling
    n = nobs(prob)
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
    p = ncovariates(prob)

    # Configure AD backend
    ad_kwargs = (;)
    if method.ad_backend == AutoReverseDiff
        ad_kwargs = merge(ad_kwargs, (compile = true,))
    elseif method.ad_backend == AutoMooncake
        @info "Using AutoMooncake. First run(s) compile differentiation rules."
    end

    # Create AD wrapper
    prob_ad = LogDensityProblemsAD.ADgradient(method.ad_backend(; ad_kwargs...), vi_model)

    # Set up variational family
    q0 = AdvancedVI.MeanFieldGaussian(zeros(d), Diagonal(fill(0.1, d)))

    # Configure algorithm
    if use_subsample
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)
        alg = AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            subsampling = subsampling,
            n_samples = method.n_montecarlo
        )
    else
        alg = AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            n_samples = method.n_montecarlo
        )
    end

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
    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    vi_type = use_subsample ? :meanfield_subsampled : :meanfield

    return BDMLVIResult(
        q_result, α_samples, α_s_samples, prob.stats,
        :basic, vi_type, n_iterations, elbo_history, converged, final_elbo
    )
end

"""
    fit(prob::BDMLHierarchicalProblem, method::UnifiedVIMethod; n_iterations=1000, n_draws=2000)

Fit hierarchical BDML model using Unified VI.
"""
function fit(
        prob::BDMLHierarchicalProblem, method::UnifiedVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create unified VI model
    vi_model = BDMLVIModel(prob.Y, prob.D, prob.X; model_type = :hier, T = Float64)

    # Determine subsampling
    n = nobs(prob)
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
    p = ncovariates(prob)

    # Configure AD
    ad_kwargs = (;)
    if method.ad_backend == AutoReverseDiff
        ad_kwargs = merge(ad_kwargs, (compile = true,))
    end

    prob_ad = LogDensityProblemsAD.ADgradient(method.ad_backend(; ad_kwargs...), vi_model)

    q0 = AdvancedVI.MeanFieldGaussian(zeros(d), Diagonal(fill(0.1, d)))

    if use_subsample
        dataset = 1:n
        subsampling = AdvancedVI.ReshufflingBatchSubsampling(dataset, batch_size)
        alg = AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            subsampling = subsampling,
            n_samples = method.n_montecarlo
        )
    else
        alg = AdvancedVI.KLMinRepGradProxDescent(
            method.ad_backend(; ad_kwargs...);
            n_samples = method.n_montecarlo
        )
    end

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

    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    vi_type = use_subsample ? :meanfield_subsampled : :meanfield

    return BDMLVIResult(
        q_result, α_samples, α_s_samples, prob.stats,
        :hier, vi_type, n_iterations, elbo_history, converged, final_elbo
    )
end

# ==================== VI DISPATCH - SIMPLE ====================

"""
    fit(prob::BDMLBasicProblem, method::SimpleVIMethod; n_iterations=1000, n_draws=2000)

Fit basic BDML model using Simple VI (Turing's native vi() function).

This works well with AutoMooncake. No subsampling support.
"""
function fit(
        prob::BDMLBasicProblem, method::SimpleVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create Turing model (uses Beta correlation for VI compatibility)
    model = bdml_basic_vi(prob.Y, prob.D, prob.X)

    # Configure AD
    ad_kwargs = (;)
    if method.ad_backend == AutoReverseDiff
        ad_kwargs = merge(ad_kwargs, (compile = true,))
    elseif method.ad_backend == AutoMooncake
        @info "Using AutoMooncake AD backend. Note: May require more compliation time, at the benefit of much faster fitting."
    end

    ad_backend = method.ad_backend(; ad_kwargs...)

    # Initialize variational distribution
    q_init = Variational.q_meanfield_gaussian(model)

    # Run VI
    q_result = vi(
        model, q_init, n_iterations;
        show_progress = show_progress,
        adtype = ad_backend
    )

    q = q_result[1]
    stats_vi = q_result[2]

    # Extract ELBO history
    elbo_history = Float64[-1.0]
    if !isempty(stats_vi) && hasproperty(stats_vi[1], :elbo)
        elbo_history = [s.elbo for s in stats_vi]
    end
    final_elbo = length(elbo_history) > 0 ? elbo_history[end] : -1.0
    converged = true

    # Draw samples
    vi_samples = rand(q, n_draws)

    # Extract alpha
    p = ncovariates(prob)
    α_s_samples = extract_alpha(vi_samples, p, :basic)

    # Transform back
    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLVIResult(
        q, α_samples, α_s_samples, prob.stats,
        :basic, :meanfield, n_iterations, elbo_history, converged, final_elbo
    )
end

"""
    fit(prob::BDMLHierarchicalProblem, method::SimpleVIMethod; n_iterations=1000, n_draws=2000)

Fit hierarchical BDML model using Simple VI.
"""
function fit(
        prob::BDMLHierarchicalProblem, method::SimpleVIMethod;
        n_iterations::Int = 1000, n_draws::Int = 2000, show_progress::Bool = true
    )

    # Create Turing model
    model = bdml_hier_vi(prob.Y, prob.D, prob.X)

    # Configure AD
    ad_kwargs = (;)
    if method.ad_backend == AutoReverseDiff
        ad_kwargs = merge(ad_kwargs, (compile = true,))
    end

    ad_backend = method.ad_backend(; ad_kwargs...)

    q_init = Variational.q_meanfield_gaussian(model)

    q_result = vi(
        model, q_init, n_iterations;
        show_progress = show_progress,
        adtype = ad_backend
    )

    q = q_result[1]
    stats_vi = q_result[2]

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

    vi_samples = rand(q, n_draws)

    # Extract alpha for hierarchical
    p = ncovariates(prob)
    α_s_samples = extract_alpha(vi_samples, p, :hier)

    scaling_factor = prob.stats.Y_sd / prob.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BDMLVIResult(
        q, α_samples, α_s_samples, prob.stats,
        :hier, :meanfield, n_iterations, elbo_history, converged, final_elbo
    )
end
