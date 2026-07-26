# VMP (Variational Message Passing) inference for BDML via RxInfer.jl.
#
# The likelihood is represented by one sufficient-statistics factor rather than
# one graph branch per observation. This keeps the RxInfer graph independent of
# n and moves the O(n p^2) work into a single BLAS-backed preprocessing step.

module BayesianDoubleMLRxInferExt

using BayesianDoubleML
using LinearAlgebra
using Random
using RxInfer

import Distributions: InverseWishart
import Statistics: mean

"""
    BDMLSufficientStatistics

Sufficient statistics for the bivariate Gaussian reduced-form likelihood.

The representation removes the observation dimension from the RxInfer graph.
"""
struct BDMLSufficientStatistics
    n::Int
    sxx::Matrix{Float64}
    xsy::Vector{Float64}
    xsd::Vector{Float64}
    sww::Matrix{Float64}
end

"""
    BDMLSufficientStatistics(model::AbstractBDMLModel)

Build the sufficient statistics used by the RxInfer SUR likelihood factor.

For standardized data ``Y``, ``D``, and ``X``, the stored quantities are
``X'X``, ``X'Y``, ``X'D``, and the 2-by-2 cross-product matrix for ``(Y, D)``.
The construction performs the only operation in this backend whose cost grows
with the number of observations.

# Arguments
- `model`: Standardized [`AbstractBDMLModel`](@ref) containing `Y`, `D`, and `X`.

# Returns
A [`BDMLSufficientStatistics`](@ref) value with concrete `Float64` arrays.
"""
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

"""
    _stats(out, sxx, xsy, xsd, sww)

Reconstruct [`BDMLSufficientStatistics`](@ref) from the point-mass values
received by the custom RxInfer factor.

# Arguments
- `out`: Observation count stored on the factor's observed output interface.
- `sxx`, `xsy`, `xsd`, `sww`: Precomputed sufficient statistics.

# Returns
A concrete sufficient-statistics container for local message updates.
"""
function _stats(out, sxx, xsy, xsd, sww)
    return BDMLSufficientStatistics(Int(out), Matrix(sxx), vec(xsy), vec(xsd), Matrix(sww))
end

"""
    _expected_residual_crossproduct(stats, qδ, qγ)

Compute the expected residual cross-product matrix
``E[(W - XB)' Ω (W - XB)]`` in its 2-by-2 residual form. The coefficient
posterior covariance terms are included in the diagonal entries, allowing
uncertainty in ``δ`` and ``γ`` to propagate into the precision update.

# Arguments
- `stats`: Sufficient statistics for the observed reduced forms.
- `qδ`, `qγ`: Current variational coefficient marginals.

# Returns
A symmetric 2-by-2 residual cross-product matrix.
"""
function _expected_residual_crossproduct(stats, qδ, qγ)
    mδ, Vδ = mean_cov(qδ)
    mγ, Vγ = mean_cov(qγ)
    Aδ = Vδ + mδ * mδ'
    Aγ = Vγ + mγ * mγ'
    r11 = stats.sww[1, 1] - 2 * dot(stats.xsy, mδ) + dot(stats.sxx, Aδ)
    r22 = stats.sww[2, 2] - 2 * dot(stats.xsd, mγ) + dot(stats.sxx, Aγ)
    r12 = stats.sww[1, 2] - dot(stats.xsy, mγ) - dot(stats.xsd, mδ) +
        dot(mδ, stats.sxx * mγ)
    return [r11 r12; r12 r22]
end

"""
    _coefficient_message(stats, qγ, qΩ, outcome::Int)

Construct the Gaussian message for one reduced-form coefficient vector. The
message uses the expected precision ``E[Ω]`` and the current mean of the other
coefficient vector, matching the parallel mean-field update used by the
manual VMP backend.

# Arguments
- `stats`: Sufficient statistics for the observed reduced forms.
- `qγ`: Marginal for the counterpart coefficient vector.
- `qΩ`: Marginal for the bivariate error precision.
- `outcome`: `1` for ``δ``/`Y` and `2` for ``γ``/`D`.

# Returns
An `MvNormalWeightedMeanPrecision` message.
"""
function _coefficient_message(stats, qγ, qΩ, outcome::Int)
    Ω = mean(qΩ)
    mγ = mean(qγ)
    if outcome == 1
        Λ = Ω[1, 1] .* stats.sxx
        ξ = Ω[1, 1] .* stats.xsy + Ω[1, 2] .* (stats.xsd - stats.sxx * mγ)
    else
        Λ = Ω[2, 2] .* stats.sxx
        ξ = Ω[2, 2] .* stats.xsd + Ω[1, 2] .* (stats.xsy - stats.sxx * mγ)
    end
    return MvNormalWeightedMeanPrecision(ξ, Symmetric(Λ))
end

"""
    _precision_message(stats, qδ, qγ)

Construct the Wishart message for the error precision ``Ω`` from the expected
residual cross-product. `WishartFast` uses the natural-parameter convention
required by ReactiveMP, so the likelihood contribution has degrees of freedom
``n + d + 1`` for ``d = 2``.

# Arguments
- `stats`: Sufficient statistics for the observed reduced forms.
- `qδ`, `qγ`: Current coefficient marginals.

# Returns
A `WishartFast` message representing the Gaussian likelihood contribution.
"""
function _precision_message(stats, qδ, qγ)
    residual = _expected_residual_crossproduct(stats, qδ, qγ)
    # `WishartFast` stores the inverse of its scale matrix internally.
    # Wishart's natural exponent is `(ν - d - 1) / 2`; the Gaussian
    # likelihood contributes `n / 2` for d = 2.
    return RxInfer.ExponentialFamily.WishartFast(stats.n + 3, Symmetric(residual))
end

"""
    BDMLSURLikelihood

Sufficient-statistics SUR likelihood factor used by the RxInfer backend.

The factor represents the bivariate reduced form
``W_i = (Y_i, D_i)' ~ N_2(B'X_i, Σ)`` while receiving only the aggregated
statistics `sxx`, `xsy`, `xsd`, and `sww`. Its message rules therefore avoid
constructing one likelihood branch, matrix multiplication node, and temporary
observation vector for every observation.

The factor exposes Gaussian messages for ``δ`` and ``γ`` and a Wishart message
for ``Ω = Σ⁻¹``. The public result converts the latter back to an
`InverseWishart` posterior for ``Σ``.
"""
struct BDMLSURLikelihood end

@node BDMLSURLikelihood Stochastic [out, sxx, xsy, xsd, sww, δ, γ, Ω]

@rule BDMLSURLikelihood(:δ, Marginalisation) (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass,
    q_sww::PointMass, q_γ::Any, q_Ω::Any,
) = _coefficient_message(
    _stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww)),
    q_γ, q_Ω, 1,
)

@rule BDMLSURLikelihood(:γ, Marginalisation) (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass,
    q_sww::PointMass, q_δ::Any, q_Ω::Any,
) = begin
    stats = _stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww))
    # Reuse the coefficient kernel with the roles of δ and γ exchanged.
    Ω = mean(q_Ω)
    mδ = mean(q_δ)
    Λ = Ω[2, 2] .* stats.sxx
    ξ = Ω[2, 2] .* stats.xsd + Ω[1, 2] .* (stats.xsy - stats.sxx * mδ)
    MvNormalWeightedMeanPrecision(ξ, Symmetric(Λ))
end

@rule BDMLSURLikelihood(:Ω, Marginalisation) (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass,
    q_sww::PointMass, q_δ::Any, q_γ::Any,
) = _precision_message(
    _stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww)),
    q_δ, q_γ,
)

@average_energy BDMLSURLikelihood (
    q_out::PointMass, q_sxx::PointMass, q_xsy::PointMass, q_xsd::PointMass,
    q_sww::PointMass, q_δ::Any, q_γ::Any, q_Ω::Any,
) = begin
    stats = _stats(mean(q_out), mean(q_sxx), mean(q_xsy), mean(q_xsd), mean(q_sww))
    residual = _expected_residual_crossproduct(stats, q_δ, q_γ)
    return (
        -stats.n * mean(logdet, q_Ω) +
            tr(mean(q_Ω) * residual) + stats.n * 2 * log(2π)
    ) / 2
end

const VMP_CONSTRAINTS_HIER = @constraints begin
    q(δ, γ, Ω) = q(δ)q(γ)q(Ω)
    q(δ, τδ) = q(δ)q(τδ)
    q(γ, τγ) = q(γ)q(τγ)
end

const VMP_CONSTRAINTS_BASIC = @constraints begin
    q(δ, γ, Ω) = q(δ)q(γ)q(Ω)
end

# Basic BDML graph with fixed coefficient priors and one sufficient-statistics
# likelihood factor.
@model function bdml_vmp_basic(n, sxx, xsy, xsd, sww, ν0, S0)
    Ω ~ Wishart(ν0, inv(S0))
    δ ~ MvNormalMeanScaleMatrixPrecision(
        μ = zeros(size(sxx, 1)), γ = 1 / 25,
        G = Matrix{Float64}(I, size(sxx, 1), size(sxx, 1))
    )
    γ ~ MvNormalMeanScaleMatrixPrecision(
        μ = zeros(size(sxx, 1)), γ = 1 / 25,
        G = Matrix{Float64}(I, size(sxx, 1), size(sxx, 1))
    )
    n ~ BDMLSURLikelihood(sxx, xsy, xsd, sww, δ, γ, Ω)
end

# Hierarchical BDML graph with independent coefficient precisions and one
# sufficient-statistics likelihood factor.
@model function bdml_vmp_hierarchical(n, sxx, xsy, xsd, sww, ν0, S0, aτ, bτ)
    Ω ~ Wishart(ν0, inv(S0))
    τδ ~ Gamma(shape = aτ, scale = bτ)
    τγ ~ Gamma(shape = aτ, scale = bτ)
    δ ~ MvNormalMeanScalePrecision(μ = zeros(size(sxx, 1)), γ = τδ)
    γ ~ MvNormalMeanScalePrecision(μ = zeros(size(sxx, 1)), γ = τγ)
    n ~ BDMLSURLikelihood(sxx, xsy, xsd, sww, δ, γ, Ω)
end

# Initial variational marginals for the hierarchical RxInfer graph.
@initialization function bdml_vmp_init_hier(p, aτ, bτ)
    q(δ) = vague(MvNormalMeanCovariance, p)
    q(γ) = vague(MvNormalMeanCovariance, p)
    μ(δ) = vague(MvNormalMeanCovariance, p)
    μ(γ) = vague(MvNormalMeanCovariance, p)
    q(Ω) = vague(Wishart, 2)
    q(τδ) = GammaShapeRate(aτ, 1.0 / bτ)
    q(τγ) = GammaShapeRate(aτ, 1.0 / bτ)
end

# Initial variational marginals for the basic RxInfer graph.
@initialization function bdml_vmp_init_basic(p)
    q(δ) = vague(MvNormalMeanCovariance, p)
    q(γ) = vague(MvNormalMeanCovariance, p)
    q(Ω) = vague(Wishart, 2)
end

"""
    _posterior_covariance(qΩ)

Convert the inferred Wishart posterior for ``Ω = Σ⁻¹`` into the public
`InverseWishart` posterior for ``Σ``.

# Arguments
- `qΩ`: Inferred Wishart precision posterior.

# Returns
An `InverseWishart` distribution with equivalent degrees of freedom and scale.
"""
function _posterior_covariance(qΩ)
    ν, scale_precision = RxInfer.ExponentialFamily.params(qΩ)
    return InverseWishart(ν, inv(Symmetric(scale_precision)))
end

"""
    _draw_alpha_samples(rng, qΩ, n_draws::Int)

Draw causal-effect samples using the BDML recovery formula
``α⁽ˢ⁾ = Σ₁₂⁽ˢ⁾ / Σ₂₂⁽ˢ⁾``.

# Arguments
- `rng`: Random number generator used for posterior draws.
- `qΩ`: Inferred Wishart precision posterior.
- `n_draws`: Number of causal-effect samples.

# Returns
A tuple `(alpha_samples_standardized, qΣ)` containing standardized effect
samples and the converted covariance posterior.
"""
function _draw_alpha_samples(rng, qΩ, n_draws::Int)
    qΣ = _posterior_covariance(qΩ)
    α = Vector{Float64}(undef, n_draws)
    for i in eachindex(α)
        Σ = rand(rng, qΣ)
        α[i] = Σ[1, 2] / Σ[2, 2]
    end
    return α, qΣ
end

"""
    _rxinfer_configuration(::BDMLBasicModel, stats, method)

Build the model-specific RxInfer configuration for the basic BDML model.

This method keeps model selection in dispatch rather than branching inside the
shared inference loop.

# Returns
A named tuple containing the model generator, constraints, initialization,
return-variable specification, and `:basic` model tag.
"""
function _rxinfer_configuration(
        ::BayesianDoubleML.BDMLBasicModel,
        stats::BDMLSufficientStatistics,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP},
    )
    p = size(stats.sxx, 1)
    S0 = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    return (
        model = bdml_vmp_basic(ν0 = method.ν0, S0 = S0),
        constraints = VMP_CONSTRAINTS_BASIC,
        initialization = bdml_vmp_init_basic(p),
        returnvars = (δ = KeepLast(), γ = KeepLast(), Ω = KeepLast()),
        model_type = :basic,
    )
end

"""
    _rxinfer_configuration(::BDMLHierarchicalModel, stats, method)

Build the model-specific RxInfer configuration for the hierarchical BDML model.

# Returns
A named tuple containing the model generator, constraints, initialization,
return-variable specification, and `:hier` model tag.
"""
function _rxinfer_configuration(
        ::BayesianDoubleML.BDMLHierarchicalModel,
        stats::BDMLSufficientStatistics,
        method::BayesianDoubleML.VMPMethod{BayesianDoubleML.RxInferVMP},
    )
    p = size(stats.sxx, 1)
    S0 = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    return (
        model = bdml_vmp_hierarchical(
            ν0 = method.ν0, S0 = S0, aτ = method.aτ, bτ = method.bτ,
        ),
        constraints = VMP_CONSTRAINTS_HIER,
        initialization = bdml_vmp_init_hier(p, method.aτ, method.bτ),
        returnvars = (
            δ = KeepLast(), γ = KeepLast(), Ω = KeepLast(),
            τδ = KeepLast(), τγ = KeepLast(),
        ),
        model_type = :hier,
    )
end

"""
    _rxinfer_posterior(result, qΣ, ::BDMLBasicModel)

Assemble the backend-neutral posterior tuple for the basic model.

# Returns
A named tuple containing posterior marginals for `δ`, `γ`, and `Σ`.
"""
function _rxinfer_posterior(
        result,
        qΣ,
        ::BayesianDoubleML.BDMLBasicModel,
    )
    return (δ = result.posteriors[:δ], γ = result.posteriors[:γ], Σ = qΣ)
end

"""
    _rxinfer_posterior(result, qΣ, ::BDMLHierarchicalModel)

Assemble the backend-neutral posterior tuple for the hierarchical model.

# Returns
A named tuple containing posterior marginals for `δ`, `γ`, `Σ`, `τ_δ`, and
`τ_γ`.
"""
function _rxinfer_posterior(
        result,
        qΣ,
        ::BayesianDoubleML.BDMLHierarchicalModel,
    )
    return (
        δ = result.posteriors[:δ], γ = result.posteriors[:γ], Σ = qΣ,
        τ_δ = result.posteriors[:τδ], τ_γ = result.posteriors[:τγ],
    )
end

"""
    _fit_vmp_rxinfer(
        model::AbstractBDMLModel,
        method::VMPMethod{RxInferVMP};
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )

Fit a BDML model with the sufficient-statistics RxInfer VMP backend.

The graph size is constant in the number of observations; preprocessing is the
only operation that scales with `n`.

# Algorithm
1. Compute `X'X`, `X'Y`, `X'D`, and the reduced-form cross-product matrix.
2. Select the basic or hierarchical graph through model-specific dispatch.
3. Run conjugate RxInfer message passing on the constant-size graph.
4. Convert the precision posterior to an `InverseWishart` covariance posterior.
5. Draw causal-effect samples using the covariance-ratio recovery formula.

# Arguments
- `model`: Standardized basic or hierarchical BDML model.
- `method`: [`VMPMethod`](@ref) using the [`RxInferVMP`](@ref) backend.

# Keyword arguments
- `n_iterations::Int = 50`: Maximum number of message-passing iterations.
- `n_draws::Int = 2000`: Number of posterior causal-effect draws.
- `rng::AbstractRNG`: Random number generator for posterior sampling.
- `show_progress::Bool = false`: Whether RxInfer displays iteration progress.

# Returns
A [`BDMLVMPResult`](@ref) with backend-neutral posterior fields and a negative
Bethe free-energy diagnostic history.
"""
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
    options = method.backend.limit_stack_depth === nothing ? nothing :
        (limit_stack_depth = method.backend.limit_stack_depth,)
    @info "BDML VMP (RxInfer): n=$(stats.n), p=$p, model_type=$(configuration.model_type), iterations=$n_iterations"

    result = infer(
        model = configuration.model,
        data = (
            n = stats.n, sxx = stats.sxx, xsy = stats.xsy, xsd = stats.xsd,
            sww = stats.sww,
        ),
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
        min_pct = 0.3,
        rel_tol = 0.05,
        check_trend = true,
        min_iterations = min(50, max(10, n_iterations ÷ 2)),
        verbose = false,
    )
    @info "VMP convergence" converged message = conv_msg n_iterations = length(bfe_history)

    α_s_samples, qΣ = _draw_alpha_samples(rng, result.posteriors[:Ω], n_draws)
    α_samples = α_s_samples .* (model.stats.Y_sd / model.stats.D_sd)
    posterior = _rxinfer_posterior(result, qΣ, model)

    return BayesianDoubleML.BDMLVMPResult(
        posterior,
        α_samples,
        α_s_samples,
        model.stats,
        configuration.model_type,
        :rxinfer,
        n_iterations,
        length(negative_bfe_history),
        negative_bfe_history,
        converged,
        final_negative_bfe,
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
