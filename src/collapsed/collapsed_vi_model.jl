# Unconstrained collapsed BDML log-density for AdvancedVI

using LogDensityProblems

struct CollapsedBDMLProblem
    stats::CollapsedBDML
    model_type::Symbol
end

function CollapsedBDMLProblem(stats::CollapsedBDML; model_type::Symbol = stats.model_type)
    model_type in (:basic, :hier) || throw(ArgumentError("model_type must be :basic or :hier"))
    model_type == stats.model_type || throw(ArgumentError("model_type must match the source model"))
    return CollapsedBDMLProblem(stats, model_type)
end

LogDensityProblems.dimension(model::CollapsedBDMLProblem) = model.model_type === :hier ? 5 : 3

# AdvancedVI must differentiate this target directly. Advertising order 1 would
# make it wrap an already AD-wrapped target and triggers AdvancedVI's Mooncake
# MixedADLogDensityProblem pullback.
LogDensityProblems.capabilities(::Type{<:CollapsedBDMLProblem}) =
    LogDensityProblems.LogDensityOrder{0}()

@inline function _log_one_minus_tanh_squared(x::Real)
    ax = abs(x)
    return 2 * (log(2) - ax - log1p(exp(-2 * ax)))
end

@inline function _scale_and_logprior(x::Real)
    shifted_scale = exp(x)
    scale = 0.1 + shifted_scale
    normalization = log(1 - (0.5 + atan(0.1 / 2.5) / pi))
    logprior = -log(pi * 2.5) - log1p((scale / 2.5)^2) - normalization + x
    return scale, logprior
end

@inline function _variance_and_logprior(x::Real)
    variance = exp(x)
    # InverseGamma(2, 2), including the exp-transform Jacobian.
    logprior = 2 * log(2) - 2 * x - 2 / variance
    return variance, logprior
end

@inline function _correlation_and_logprior(x::Real)
    rho = tanh(x)
    log_one_minus_rho2 = _log_one_minus_tanh_squared(x)
    # rho_raw=(rho+1)/2 ~ Beta(2,2), including d(rho)/dx and d(rho_raw)/d(rho).
    return rho, log(3 / 4) + 2 * log_one_minus_rho2
end

function LogDensityProblems.logdensity(model::CollapsedBDMLProblem, theta)
    sigma_U, lp_U = _scale_and_logprior(theta[1])
    sigma_V, lp_V = _scale_and_logprior(theta[2])
    rho, lp_rho = _correlation_and_logprior(theta[3])
    logprior = lp_U + lp_V + lp_rho

    if model.model_type === :hier
        s2d, lp_s2d = _variance_and_logprior(theta[4])
        s2g, lp_s2g = _variance_and_logprior(theta[5])
        logprior += lp_s2d + lp_s2g
    else
        s2d = 25.0
        s2g = 25.0
    end

    loglikelihood = log_marginal_collapsed(
        model.stats, sigma_U, sigma_V, rho, s2d, s2g,
    )
    value = logprior + loglikelihood
    return isfinite(value) ? value : -Inf
end

@inline function unpack_collapsed(model::CollapsedBDMLProblem, theta)
    sigma_U = 0.1 + exp(theta[1])
    sigma_V = 0.1 + exp(theta[2])
    rho = tanh(theta[3])
    if model.model_type === :hier
        return sigma_U, sigma_V, rho, exp(theta[4]), exp(theta[5])
    end
    return sigma_U, sigma_V, rho, 25.0, 25.0
end
