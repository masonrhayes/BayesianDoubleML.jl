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
#
# The bivariate mean of W_i is assembled without stacking random scalars by
# using constant 2×p design matrices:
#   Cδ[i] = [x_i'; 0'],  Cγ[i] = [0'; x_i']  =>  Cδ[i]·δ + Cγ[i]·γ = [x_i'δ; x_i'γ]
# This uses only verified rules: matrix(PointMass) × vector(MvNormal) products
# and multivariate normal addition.
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

# Mean-field constraints across blocks (within-block structure preserved:
# q(δ), q(γ) full-rank MvNormal, q(Σ) full Inverse-Wishart)
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

# Initialization (breaks VMP dependency loops; follows RxInfer's iid-covariance idiom)
@initialization function bdml_vmp_init_hier(p)
    q(δ) = vague(MvNormalMeanCovariance, p)
    q(γ) = vague(MvNormalMeanCovariance, p)
    μ(δ) = vague(MvNormalMeanCovariance, p)
    μ(γ) = vague(MvNormalMeanCovariance, p)
    q(Σ) = vague(InverseWishart, 2)
    q(ηδ) = vague(MvNormalMeanCovariance, 2)
    q(ηγ) = vague(MvNormalMeanCovariance, 2)
    q(μ) = vague(MvNormalMeanCovariance, 2)
    q(τ_δ) = GammaShapeRate(2.0, 2.0)
    q(τ_γ) = GammaShapeRate(2.0, 2.0)
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

"""
    _vmp_design_mats(x::AbstractVector)

Build the constant 2×p design matrices for one observation such that
Cδ·δ + Cγ·γ = [x'δ; x'γ].
"""
function _vmp_design_mats(x::AbstractVector{Float64})
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

function _fit_vmp(
        model::BayesianDoubleML.AbstractBDMLModel, model_type::Symbol;
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        ν0::Float64 = 4.0,
        S0::Union{Nothing, AbstractMatrix} = nothing,
        aτ::Float64 = 2.0,
        bτ::Float64 = 0.5,
        showprogress::Bool = false,
        seed::Union{Nothing, Integer} = nothing,
        limit_stack_depth::Union{Nothing, Int} = nothing,
    )
    n = BayesianDoubleML.nobs(model)
    p = BayesianDoubleML.ncovariates(model)
    hierarchical = model_type === :hier

    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    ν0 > 3 || throw(ArgumentError("ν0 must exceed 3 so the 2×2 Inverse-Wishart prior has a finite mean"))
    aτ > 0 || throw(ArgumentError("aτ must be positive"))
    bτ > 0 || throw(ArgumentError("bτ must be positive"))
    limit_stack_depth === nothing || limit_stack_depth > 0 || throw(ArgumentError("limit_stack_depth must be positive"))

    # Auto-enable stack depth limiting for very large models to prevent StackOverflowError
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

    S0_mat = S0 === nothing ? Matrix{Float64}(I, 2, 2) : Matrix{Float64}(S0)
    size(S0_mat) == (2, 2) || throw(ArgumentError("S0 must be a 2×2 matrix"))
    issymmetric(S0_mat) || throw(ArgumentError("S0 must be symmetric"))
    isposdef(Symmetric(S0_mat)) || throw(ArgumentError("S0 must be positive definite"))

    constraints = hierarchical ? vmp_constraints_hier : vmp_constraints_basic
    init = hierarchical ? bdml_vmp_init_hier(p) : bdml_vmp_init_basic(p)
    returnvars = if hierarchical
        (δ = KeepLast(), γ = KeepLast(), Σ = KeepLast(), τ_δ = KeepLast(), τ_γ = KeepLast())
    else
        (δ = KeepLast(), γ = KeepLast(), Σ = KeepLast())
    end

    @info "BDML VMP (RxInfer): n=$n, p=$p, model_type=$model_type, iterations=$n_iterations"

    options = limit_stack_depth === nothing ? nothing : (limit_stack_depth = limit_stack_depth,)

    # ReactiveMP's Gaussian product messages can be microscopically
    # non-symmetric before FastCholesky symmetrizes them internally. Suppress
    # only that documented numerical warning; all RxInfer warnings remain.
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
            showprogress = showprogress,
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
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    α_s_samples = Vector{Float64}(undef, n_draws)
    for s in 1:n_draws
        Σ_s = rand(rng, qΣ)
        α_s_samples[s] = Σ_s[1, 2] / Σ_s[2, 2]
    end

    # Transform back to original scale
    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    return BayesianDoubleML.BDMLVIResult(
        result.posteriors, α_samples, α_s_samples, model.stats,
        model_type, :structured, :vmp, n_iterations, elbo_history, converged, final_elbo
    )
end

# Dispatch implementations (more specific than the informative-error fallback in fit.jl)
function BayesianDoubleML._fit_impl(model::BayesianDoubleML.BDMLBasicModel, method::BayesianDoubleML.VMPMethod; kwargs...)
    return _fit_vmp(model, :basic; kwargs...)
end

function BayesianDoubleML._fit_impl(model::BayesianDoubleML.BDMLHierarchicalModel, method::BayesianDoubleML.VMPMethod; kwargs...)
    return _fit_vmp(model, :hier; kwargs...)
end

end # module
