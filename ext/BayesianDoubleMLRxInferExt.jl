# VMP via RxInfer with structured joint q(beta) where beta=[delta;gamma]
# Retains Cov(delta,gamma) via joint 2p Gaussian (block 2x2 in eigenbasis is aggregated into dense 2p covariance)
# For hierarchical, uses custom prior factor BDMLHierJointPrior to keep q(beta)q(tau) with exact tau updates.

module BayesianDoubleMLRxInferExt

using BayesianDoubleML
using LinearAlgebra
using Random
using RxInfer

import Distributions: InverseWishart
import Statistics: mean

struct BDMLSufficientStatistics
    n::Int
    sxx::Matrix{Float64}
    xsy::Vector{Float64}
    xsd::Vector{Float64}
    sww::Matrix{Float64}
end

function BDMLSufficientStatistics(model::BayesianDoubleML.AbstractBDMLModel)
    y = model.Y
    d = model.D
    x = model.X
    yd = dot(y, d)
    return BDMLSufficientStatistics(
        length(y),
        Matrix(Symmetric(x' * x)),
        x' * y,
        x' * d,
        [dot(y, y) yd; yd dot(d, d)],
    )
end

function _stats(out, sxx, xsy, xsd, sww)
    return BDMLSufficientStatistics(Int(out), Matrix(sxx), vec(xsy), vec(xsd), Matrix(sww))
end

# ---------- Joint likelihood factor ----------

function _joint_expected_residual(stats, qβ)
    m, V = mean_cov(qβ)
    p = length(stats.xsy)
    mδ = m[1:p]; mγ = m[(p + 1):2p]
    Vδ = V[1:p, 1:p]; Vγ = V[(p + 1):2p, (p + 1):2p]; C = V[1:p, (p + 1):2p]
    Aδ = Vδ + mδ * mδ'
    Aγ = Vγ + mγ * mγ'
    r11 = stats.sww[1, 1] - 2 * dot(stats.xsy, mδ) + dot(stats.sxx, Aδ)
    r22 = stats.sww[2, 2] - 2 * dot(stats.xsd, mγ) + dot(stats.sxx, Aγ)
    r12 = stats.sww[1, 2] - dot(stats.xsy, mγ) - dot(stats.xsd, mδ) + dot(mδ, stats.sxx * mγ) + tr(stats.sxx * C)
    return [r11 r12; r12 r22]
end

function _joint_coeff_message(stats, qΩ)
    Ω = mean(qΩ)
    p = size(stats.sxx, 1)
    XW = hcat(stats.xsy, stats.xsd) # p×2
    XWΩ = XW * Ω # p×2
    ξ = vec(XWΩ) # 2p
    Λ = zeros(2p, 2p)
    Λ[1:p, 1:p] = Ω[1, 1] * stats.sxx
    Λ[(p + 1):2p, (p + 1):2p] = Ω[2, 2] * stats.sxx
    Λ[1:p, (p + 1):2p] = Ω[1, 2] * stats.sxx
    Λ[(p + 1):2p, 1:p] = Ω[1, 2] * stats.sxx
    return MvNormalWeightedMeanPrecision(ξ, Symmetric(Λ))
end

function _joint_prec_message(stats, qβ)
    residual = _joint_expected_residual(stats, qβ)
    return RxInfer.ExponentialFamily.WishartFast(stats.n + 3, Symmetric(residual))
end

struct BDMLJointLikelihood end
@node BDMLJointLikelihood Stochastic [out, sxx, xsy, xsd, sww, β, Ω]

@rule BDMLJointLikelihood(:β, Marginalisation) (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass, q_sww::PointMass, q_Ω::Any,
) = _joint_coeff_message(_stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww)), q_Ω)

@rule BDMLJointLikelihood(:Ω, Marginalisation) (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass, q_sww::PointMass, q_β::Any,
) = _joint_prec_message(_stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww)), q_β)

@average_energy BDMLJointLikelihood (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass, q_sww::PointMass, q_β::Any, q_Ω::Any,
) = begin
    stats = _stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww))
    residual = _joint_expected_residual(stats, q_β)
    return (-stats.n * mean(logdet, q_Ω) + tr(mean(q_Ω) * residual) + stats.n * 2 * log(2π)) / 2
end

# ---------- Hierarchical joint prior factor ----------

struct BDMLHierJointPrior end
@node BDMLHierJointPrior Stochastic [out, τδ, τγ, sxx]

@rule BDMLHierJointPrior(:out, Marginalisation) (
    q_τδ::Any, q_τγ::Any, q_sxx::PointMass,
) = begin
    p = size(mean(q_sxx), 1)
    λδ = mean(q_τδ)
    λγ = mean(q_τγ)
    Λ = zeros(2p, 2p)
    Λ[1:p, 1:p] = λδ * Matrix(I, p, p)
    Λ[(p + 1):2p, (p + 1):2p] = λγ * Matrix(I, p, p)
    return MvNormalWeightedMeanPrecision(zeros(2p), Symmetric(Λ))
end

# Likelihood contribution to τ (without prior a,b) — prior Gamma will combine via q(τ)=q(τ_prior)q(τ_lik)
@rule BDMLHierJointPrior(:τδ, Marginalisation) (
    q_out::Any, q_τγ::Any, q_sxx::PointMass,
) = begin
    p = size(mean(q_sxx), 1)
    m, V = mean_cov(q_out)
    mδ = m[1:p]; Vδ = V[1:p, 1:p]
    # Gamma(p/2, 0.5*Q) in shape-rate
    shape = p / 2
    rate = 0.5 * (dot(mδ, mδ) + tr(Vδ))
    # Return Gamma with shape/rate; Use GammaShapeRate for variational
    return GammaShapeRate(shape, rate)
end

@rule BDMLHierJointPrior(:τγ, Marginalisation) (
    q_out::Any, q_τδ::Any, q_sxx::PointMass,
) = begin
    p = size(mean(q_sxx), 1)
    m, V = mean_cov(q_out)
    mγ = m[(p + 1):2p]; Vγ = V[(p + 1):2p, (p + 1):2p]
    shape = p / 2
    rate = 0.5 * (dot(mγ, mγ) + tr(Vγ))
    return GammaShapeRate(shape, rate)
end

@average_energy BDMLHierJointPrior (
    q_out::Any, q_τδ::Any, q_τγ::Any, q_sxx::PointMass,
) = begin
    p = size(mean(q_sxx), 1)
    m, V = mean_cov(q_out)
    mδ = m[1:p]; mγ = m[(p + 1):2p]
    Vδ = V[1:p, 1:p]; Vγ = V[(p + 1):2p, (p + 1):2p]
    Elogτδ = mean(log, q_τδ); Eτδ = mean(q_τδ)
    Elogτγ = mean(log, q_τγ); Eτγ = mean(q_τγ)
    Qδ = dot(mδ, mδ) + tr(Vδ); Qγ = dot(mγ, mγ) + tr(Vγ)
    # E[log p(β|τ)] = -0.5*( 2p log2π - p Elogτδ - p Elogτγ + Eτδ Qδ + Eτγ Qγ )
    # This is the expected log prior of β given τ, needed for free energy
    return 0.5 * (2p * log(2π) - p * Elogτδ - p * Elogτγ + Eτδ * Qδ + Eτγ * Qγ)
end

# Constraints: joint q(beta) not factorized
const VMP_CONSTRAINTS_BASIC_JOINT = @constraints begin
    q(β, Ω) = q(β)q(Ω)
end

const VMP_CONSTRAINTS_HIER_JOINT = @constraints begin
    q(β, Ω) = q(β)q(Ω)
    q(β, τδ) = q(β)q(τδ)
    q(β, τγ) = q(β)q(τγ)
    q(τδ, τγ) = q(τδ)q(τγ)
end

@model function bdml_vmp_basic_joint(n, sxx, xsy, xsd, sww, ν0, S0)
    Ω ~ Wishart(ν0, inv(S0))
    β ~ MvNormalMeanCovariance(zeros(2 * size(sxx, 1)), 25.0 * Matrix(I, 2 * size(sxx, 1), 2 * size(sxx, 1)))
    n ~ BDMLJointLikelihood(sxx, xsy, xsd, sww, β, Ω)
end

@model function bdml_vmp_hier_joint(n, sxx, xsy, xsd, sww, ν0, S0, aτ, bτ)
    Ω ~ Wishart(ν0, inv(S0))
    τδ ~ GammaShapeScale(aτ, bτ)
    τγ ~ GammaShapeScale(aτ, bτ)
    β ~ BDMLHierJointPrior(τδ, τγ, sxx)
    n ~ BDMLJointLikelihood(sxx, xsy, xsd, sww, β, Ω)
end

@initialization function bdml_vmp_init_basic_joint(p)
    q(β) = vague(MvNormalMeanCovariance, 2p)
    μ(β) = vague(MvNormalMeanCovariance, 2p)
    q(Ω) = vague(Wishart, 2)
end

@initialization function bdml_vmp_init_hier_joint(p, aτ, bτ)
    q(β) = vague(MvNormalMeanCovariance, 2p)
    μ(β) = vague(MvNormalMeanCovariance, 2p)
    q(Ω) = vague(Wishart, 2)
    q(τδ) = GammaShapeRate(aτ, 1.0 / bτ)
    q(τγ) = GammaShapeRate(aτ, 1.0 / bτ)
end

function _posterior_covariance(qΩ)
    ν, scale_precision = RxInfer.ExponentialFamily.params(qΩ)
    return InverseWishart(ν, inv(Symmetric(scale_precision)))
end

function _draw_alpha_samples(rng, qΩ, n_draws::Int)
    qΣ = _posterior_covariance(qΩ)
    α = Vector{Float64}(undef, n_draws)
    for i in eachindex(α)
        Σ = rand(rng, qΣ)
        α[i] = Σ[1, 2] / Σ[2, 2]
    end
    return α, qΣ
end

function _rxinfer_configuration(
        ::BayesianDoubleML.BDMLBasicModel,
        stats::BDMLSufficientStatistics,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP},
    )
    p = size(stats.sxx, 1)
    S0 = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    return (
        model = bdml_vmp_basic_joint(ν0 = method.ν0, S0 = S0),
        constraints = VMP_CONSTRAINTS_BASIC_JOINT,
        initialization = bdml_vmp_init_basic_joint(p),
        returnvars = (β = KeepLast(), Ω = KeepLast()),
        model_type = :basic,
        p = p,
    )
end

function _rxinfer_configuration(
        ::BayesianDoubleML.BDMLHierarchicalModel,
        stats::BDMLSufficientStatistics,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP},
    )
    p = size(stats.sxx, 1)
    S0 = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    return (
        model = bdml_vmp_hier_joint(ν0 = method.ν0, S0 = S0, aτ = method.aτ, bτ = method.bτ),
        constraints = VMP_CONSTRAINTS_HIER_JOINT,
        initialization = bdml_vmp_init_hier_joint(p, method.aτ, method.bτ),
        returnvars = (β = KeepLast(), Ω = KeepLast(), τδ = KeepLast(), τγ = KeepLast()),
        model_type = :hier,
        p = p,
    )
end

function _rxinfer_posterior(
        result,
        qΣ,
        ::BayesianDoubleML.BDMLBasicModel,
        p::Int,
    )
    qβ = result.posteriors[:β]
    m, V = mean_cov(qβ)
    mδ = m[1:p]; mγ = m[(p + 1):2p]
    Vδ = V[1:p, 1:p]; Vγ = V[(p + 1):2p, (p + 1):2p]
    return (
        δ = MvNormal(mδ, Symmetric(Vδ)),
        γ = MvNormal(mγ, Symmetric(Vγ)),
        Σ = qΣ,
        β = qβ,
    )
end

function _rxinfer_posterior(
        result,
        qΣ,
        ::BayesianDoubleML.BDMLHierarchicalModel,
        p::Int,
    )
    qβ = result.posteriors[:β]
    m, V = mean_cov(qβ)
    mδ = m[1:p]; mγ = m[(p + 1):2p]
    Vδ = V[1:p, 1:p]; Vγ = V[(p + 1):2p, (p + 1):2p]
    return (
        δ = MvNormal(mδ, Symmetric(Vδ)),
        γ = MvNormal(mγ, Symmetric(Vγ)),
        Σ = qΣ,
        τ_δ = result.posteriors[:τδ],
        τ_γ = result.posteriors[:τγ],
        β = qβ,
    )
end

function _fit_vmp_rxinfer(
        model::BayesianDoubleML.AbstractBDMLModel,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP};
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )
    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    stats = BDMLSufficientStatistics(model)
    p = size(stats.sxx, 1)
    configuration = _rxinfer_configuration(model, stats, method)
    options = method.backend.limit_stack_depth === nothing ? nothing : (limit_stack_depth = method.backend.limit_stack_depth,)
    @info "BDML VMP (RxInfer joint): n=$(stats.n), p=$p, model_type=$(configuration.model_type), iterations=$n_iterations"
    result = infer(
        model = configuration.model,
        data = (n = stats.n, sxx = stats.sxx, xsy = stats.xsy, xsd = stats.xsd, sww = stats.sww),
        constraints = configuration.constraints,
        initialization = configuration.initialization,
        iterations = n_iterations,
        returnvars = configuration.returnvars,
        free_energy = Float64,
        showprogress = show_progress,
        options = options,
    )
    bfe_history = Vector{Float64}(result.free_energy)
    negative_bfe_history = -bfe_history
    final_negative_bfe = isempty(negative_bfe_history) ? -Inf : negative_bfe_history[end]
    converged, conv_msg = BayesianDoubleML.check_elbo_convergence(
        negative_bfe_history;
        min_pct = 0.3, rel_tol = 0.05, check_trend = true,
        min_iterations = min(50, max(10, n_iterations ÷ 2)), verbose = false,
    )
    @info "VMP convergence" converged message = conv_msg n_iterations = length(bfe_history)
    α_s_samples, qΣ = _draw_alpha_samples(rng, result.posteriors[:Ω], n_draws)
    α_samples = α_s_samples .* (model.stats.Y_sd / model.stats.D_sd)
    posterior = _rxinfer_posterior(result, qΣ, model, p)
    return BayesianDoubleML.BDMLVMPResult(
        posterior, α_samples, α_s_samples, model.stats,
        configuration.model_type, :rxinfer,
        n_iterations, length(negative_bfe_history),
        negative_bfe_history, converged, final_negative_bfe,
        :negative_bethe_free_energy,
    )
end

function BayesianDoubleML._fit_vmp(
        model::BayesianDoubleML.BDMLBasicModel,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP};
        kwargs...
    )
    return _fit_vmp_rxinfer(model, method; kwargs...)
end

function BayesianDoubleML._fit_vmp(
        model::BayesianDoubleML.BDMLHierarchicalModel,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP};
        kwargs...
    )
    return _fit_vmp_rxinfer(model, method; kwargs...)
end

end # module
