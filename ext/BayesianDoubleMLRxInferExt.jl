# VMP (Variational Message Passing) inference for BDML via RxInfer.jl
#
# This extension is loaded automatically when the user runs `using RxInfer`.
# It implements a *conjugate reparameterization* of the BDML reduced form
# (DiTraglia & Liu 2025, Section 4):
#
#   W_i = (Y_i, D_i)' ~ N_2(B'x_i, Σ),   B = [δ γ]
#
# with the conditionally conjugate SUR prior of the paper's asymptotic theory
# (Section 5, Eq. 19):
#
#   Σ ~ InverseWishart(ν₀, S₀)          [replaces LKJ(4) + Half-Cauchy]
#   δ ~ N(0, τ_δ⁻¹ I),  τ_δ ~ Gamma(2, 1/2)   [hierarchical variant]
#   γ ~ N(0, τ_γ⁻¹ I),  τ_γ ~ Gamma(2, 1/2)
#
# Gamma(shape=2, scale=1/2) on a precision is exactly InvGamma(2, 2) on the
# corresponding variance, so the paper's Student-t(4) interpretation of the
# hierarchical shrinkage prior is preserved.
#
# All message updates are closed-form (verified against ReactiveMP rules):
# q*(Σ) is Inverse-Wishart, q*(δ), q*(γ) are full-rank MvNormals, q*(τ) Gamma.
# The causal effect is recovered as in Algorithm 1, Eq. 15:
#   α⁽ˢ⁾ = Σ₁₂⁽ˢ⁾ / Σ₂₂⁽ˢ⁾,  Σ⁽ˢ⁾ ~ q*(Σ)
# Because the q*(Σ) scale matrix accumulates E_q[residual outer products]
# (including the posterior covariance of the coefficients), coefficient
# uncertainty propagates into the α posterior.

module BayesianDoubleMLRxInferExt

using BayesianDoubleML
using RxInfer
using LinearAlgebra
using Statistics
using Random


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

@model function bdml_vmp(W, Cδ, Cγ, p, hierarchical, ν0, S0, aτ, bτ)
    Σ ~ InverseWishart(ν0, S0)
    if hierarchical
        τ_δ ~ Gamma(shape = aτ, scale = bτ)
        τ_γ ~ Gamma(shape = aτ, scale = bτ)
        δ ~ MvNormalMeanScalePrecision(μ = zeros(p), γ = τ_δ)
        γ ~ MvNormalMeanScalePrecision(μ = zeros(p), γ = τ_γ)
    else
        δ ~ MvNormal(mean = zeros(p), covariance = 25.0 * Matrix{Float64}(I, p, p))
        γ ~ MvNormal(mean = zeros(p), covariance = 25.0 * Matrix{Float64}(I, p, p))
    end
    for i in eachindex(W)
        ηδ[i] := Cδ[i] * δ
        ηγ[i] := Cγ[i] * γ
        μ[i] := ηδ[i] + ηγ[i]
        W[i] ~ MvNormal(mean = μ[i], covariance = Σ)
    end
end

const vmp_constraints_hier = @constraints begin
    q(μ, Σ) = q(μ)q(Σ)
    q(ηδ, ηγ) = q(ηδ)q(ηγ)
    q(δ, τ_δ) = q(δ)q(τ_δ)
    q(γ, τ_γ) = q(γ)q(τ_γ)
end

const vmp_constraints_basic = @constraints begin
    q(μ, Σ) = q(μ)q(Σ)
    q(ηδ, ηγ) = q(ηδ)q(ηγ)
end

@initialization function bdml_vmp_init_hier(p, aτ, bτ)
    q(δ) = vague(MvNormalMeanCovariance, p)
    q(γ) = vague(MvNormalMeanCovariance, p)
    μ(δ) = vague(MvNormalMeanCovariance, p)
    μ(γ) = vague(MvNormalMeanCovariance, p)
    q(Σ) = vague(InverseWishart, 2)
    q(ηδ) = vague(MvNormalMeanCovariance, 2)
    q(ηγ) = vague(MvNormalMeanCovariance, 2)
    q(μ) = vague(MvNormalMeanCovariance, 2)
    q(τ_δ) = GammaShapeRate(aτ, 1.0 / bτ)
    q(τ_γ) = GammaShapeRate(aτ, 1.0 / bτ)
end

@initialization function bdml_vmp_init_basic(p)
    q(δ) = vague(MvNormalMeanCovariance, p)
    q(γ) = vague(MvNormalMeanCovariance, p)
    μ(δ) = vague(MvNormalMeanCovariance, p)
    μ(γ) = vague(MvNormalMeanCovariance, p)
    q(Σ) = vague(InverseWishart, 2)
    q(ηδ) = vague(MvNormalMeanCovariance, 2)
    q(ηγ) = vague(MvNormalMeanCovariance, 2)
    q(μ) = vague(MvNormalMeanCovariance, 2)
end

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

function _vmp_design_mats(x::AbstractVector{<:Real})
    p = length(x)
    Cδ = zeros(2, p)
    Cδ[1, :] .= x
    Cγ = zeros(2, p)
    Cγ[2, :] .= x
    return Cδ, Cγ
end

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

function BayesianDoubleML._fit_vmp(
        ::BayesianDoubleML.RxInferVMP,
        model::BayesianDoubleML.AbstractBDMLModel,
        method::BayesianDoubleML.VMPMethod;
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )
    hierarchical = model isa BayesianDoubleML.BDMLHierarchicalModel
    n = BayesianDoubleML.nobs(model)
    p = BayesianDoubleML.ncovariates(model)

    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))

    ν0 = method.ν0
    S0_mat = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    aτ = method.aτ
    bτ = method.bτ
    limit_stack_depth = method.backend.limit_stack_depth

    # Auto-enable stack depth limiting for very large models
    if limit_stack_depth === nothing && n > 2000
        limit_stack_depth = 200
    end

    # Bivariate observations W_i = (Y_i, D_i) and constant design matrices
    W = [[model.Y[i], model.D[i]] for i in 1:n]
    Cδ = Vector{Matrix{Float64}}(undef, n)
    Cγ = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        Cδ[i], Cγ[i] = _vmp_design_mats(vec(model.X[i, :]))
    end

    constraints = hierarchical ? vmp_constraints_hier : vmp_constraints_basic
    init = hierarchical ? bdml_vmp_init_hier(p, aτ, bτ) : bdml_vmp_init_basic(p)
    returnvars = if hierarchical
        (δ = KeepLast(), γ = KeepLast(), Σ = KeepLast(), τ_δ = KeepLast(), τ_γ = KeepLast())
    else
        (δ = KeepLast(), γ = KeepLast(), Σ = KeepLast())
    end

    @info "BDML VMP (RxInfer): n=$n, p=$p, model_type=$(hierarchical ? :hier : :basic), iterations=$n_iterations"

    options = limit_stack_depth === nothing ? nothing : (limit_stack_depth = limit_stack_depth,)

    # TODO: withenv modifies process-wide ENV and is not thread-safe.
    # Replace with a task-local or library-level toggle when available.
    result = withenv("JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC" => "1") do
        infer(
            model = bdml_vmp(
                p = p, hierarchical = hierarchical, ν0 = ν0, S0 = S0_mat, aτ = aτ, bτ = bτ
            ),
            data = (W = W, Cδ = Cδ, Cγ = Cγ),
            constraints = constraints,
            initialization = init,
            iterations = n_iterations,
            returnvars = returnvars,
            free_energy = Float64,
            showprogress = show_progress,
            options = options,
        )
    end

    # Bethe Free Energy decreases to a minimum; ELBO = -BFE
    bfe_history = Float64.(collect(result.free_energy))
    elbo_history = -bfe_history
    final_elbo = isempty(elbo_history) ? -Inf : elbo_history[end]

    converged, conv_msg = BayesianDoubleML.check_elbo_convergence(
        elbo_history;
        min_pct = 0.3,
        rel_tol = 0.05,
        check_trend = true,
        min_iterations = min(50, max(10, n_iterations ÷ 2)),
        verbose = false,
    )
    @info "VMP convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    # Causal effect posterior (Algorithm 1, Eq. 15): α⁽ˢ⁾ = Σ₁₂⁽ˢ⁾/Σ₂₂⁽ˢ⁾
    qΣ = result.posteriors[:Σ]
    α_s_samples = Vector{Float64}(undef, n_draws)
    for s in 1:n_draws
        Σ_s = rand(rng, qΣ)
        α_s_samples[s] = Σ_s[1, 2] / Σ_s[2, 2]
    end

    # Transform back to original scale
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    posterior = hierarchical ?
        (
            δ = result.posteriors[:δ],
            γ = result.posteriors[:γ],
            Σ = result.posteriors[:Σ],
            τ_δ = result.posteriors[:τ_δ],
            τ_γ = result.posteriors[:τ_γ],
        ) :
        (
            δ = result.posteriors[:δ],
            γ = result.posteriors[:γ],
            Σ = result.posteriors[:Σ],
        )

    return BayesianDoubleML.BDMLVMPResult(
        posterior,
        α_samples,
        α_s_samples,
        model.stats,
        hierarchical ? :hier : :basic,
        :rxinfer,
        n_iterations,
        length(elbo_history),
        elbo_history,
        converged,
        final_elbo,
        :elbo,
    )
end

end # module
