# Coeftable implementation following StatsAPI conventions
# Provides statistical summaries for BDML results with HPD intervals

using StatsAPI

# Helper function for formatting numbers with fixed width
to_s(x, width = 12, digits = 4) = lpad(round(x, digits = digits), width)

"""
    hpd_interval(samples::Vector{Float64}; level::Real=0.95)

Compute the Highest Posterior Density (HPD) interval for a vector of samples.
HPD intervals are the shortest intervals containing the specified probability mass.
"""
function hpd_interval(samples::Vector{Float64}; level::Real = 0.95)
    n = length(samples)
    sorted_samples = sort(samples)

    # Number of samples in the interval
    interval_n = max(1, floor(Int, level * n))

    # Find the shortest interval
    min_width = Inf
    hpd_lower = sorted_samples[1]
    hpd_upper = sorted_samples[interval_n]

    for i in 1:(n - interval_n + 1)
        width = sorted_samples[i + interval_n - 1] - sorted_samples[i]
        if width < min_width
            min_width = width
            hpd_lower = sorted_samples[i]
            hpd_upper = sorted_samples[i + interval_n - 1]
        end
    end

    return (lower = hpd_lower, upper = hpd_upper)
end

"""
    compute_pvalue(samples::Vector{Float64})

Compute two-sided p-value for testing H0: α = 0.
Uses P(α > 0) and P(α < 0) from the posterior.
"""
function compute_pvalue(samples::Vector{Float64})
    n = length(samples)

    # Count samples on each side of zero
    n_positive = sum(samples .> 0)
    n_negative = sum(samples .< 0)
    n_zero = n - n_positive - n_negative

    # Compute proportions (including half of zeros on each side)
    p_positive = (n_positive + 0.5 * n_zero) / n
    p_negative = (n_negative + 0.5 * n_zero) / n

    # Two-sided p-value
    pval = 2 * min(p_positive, p_negative)

    # Cap at 1.0
    return min(pval, 1.0)
end

# Coeftable struct - follows StatsAPI conventions but doesn't inherit from RegressionTable
# (RegressionTable is not available in all StatsAPI versions)
struct BDMLCoeftable
    coefnames::Vector{String}
    coef::Vector{Float64}
    stderror::Vector{Float64}
    mcse::Vector{Float64}
    cilower::Vector{Float64}
    ciupper::Vector{Float64}
    pvalue::Vector{Float64}
    ess::Vector{Float64}
    elbo::Union{Float64, Nothing}
    level::Float64
    nsamples::Int
    model_type::Symbol
    method_type::Symbol
end

"""
    coeftable(result::BDMLMCMCResult; level=0.95)

Compute coefficient table for MCMC results with HPD credible intervals.
Uses MCMCChains for ESS and MCSE calculations.
"""
function coeftable(result::BDMLMCMCResult; level = 0.95)
    samples = result.alpha_samples

    # Basic statistics
    coef_est = mean(samples)
    std_error = std(samples)

    # HPD interval
    hpd = hpd_interval(samples; level = level)

    # P-value
    pval = compute_pvalue(samples)

    # ESS from MCMCChains (get minimum across all parameters as conservative estimate)
    ess_df = MCMCChains.ess(result.chain)
    ess_values = ess_df.nt.ess
    finite_ess = filter(x -> isfinite(x) && x > 0, ess_values)
    ess_val = length(finite_ess) > 0 ? minimum(finite_ess) : 0.0

    # MCSE from MCMCChains (get maximum across all parameters as conservative estimate)
    mcse_df = MCMCChains.mcse(result.chain)
    mcse_values = mcse_df.nt.mcse
    finite_mcse = filter(x -> isfinite(x) && x > 0, mcse_values)
    mcse_val = length(finite_mcse) > 0 ? maximum(finite_mcse) : std(samples) / sqrt(length(samples))

    # For MCMC, no ELBO
    elbo_val = nothing

    return BDMLCoeftable(
        ["α"],
        [coef_est],
        [std_error],
        [mcse_val],
        [hpd.lower],
        [hpd.upper],
        [pval],
        [ess_val],
        elbo_val,
        level,
        length(samples),
        result.model_type,
        :MCMC
    )
end

"""
    coeftable(result::BDMLVIResult; level=0.95)

Compute coefficient table for VI results with HPD credible intervals.
Note: ESS and MCSE are not computed for VI results as samples are drawn independently.
"""
function coeftable(result::BDMLVIResult; level = 0.95)
    samples = result.alpha_samples

    # Basic statistics
    coef_est = mean(samples)
    std_error = std(samples)

    # HPD interval
    hpd = hpd_interval(samples; level = level)

    # P-value
    pval = compute_pvalue(samples)

    # For VI, no ESS or MCSE (samples are independent)
    ess_val = 0.0
    mcse_val = 0.0

    # ELBO from VI
    elbo_val = result.final_elbo

    return BDMLCoeftable(
        ["α"],
        [coef_est],
        [std_error],
        [mcse_val],
        [hpd.lower],
        [hpd.upper],
        [pval],
        [ess_val],
        elbo_val,
        level,
        length(samples),
        result.model_type,
        :VI
    )
end

# Allow generic AbstractBDMLResult dispatch
function coeftable(result::AbstractBDMLResult; level = 0.95)
    if result isa BDMLMCMCResult
        return coeftable(result::BDMLMCMCResult; level = level)
    elseif result isa BDMLVIResult
        return coeftable(result::BDMLVIResult; level = level)
    else
        error("Unknown result type: $(typeof(result))")
    end
end

# StatsAPI interface functions
StatsAPI.coefnames(ct::BDMLCoeftable) = ct.coefnames
StatsAPI.coef(ct::BDMLCoeftable) = ct.coef
StatsAPI.stderror(ct::BDMLCoeftable) = ct.stderror

# Pretty printing for BDMLCoeftable
function Base.show(io::IO, ct::BDMLCoeftable)
    println(io, "Bayesian Double ML Coefficient Table")
    println(io, "="^70)
    println(io, "Parameter: α (treatment effect)")
    println(io, "Model type: $(ct.model_type)")
    println(io, "Inference method: $(ct.method_type)")
    println(io, "Credible interval level: $(round(ct.level * 100, digits = 1))% (HPD)")
    println(io, "Number of posterior samples: $(ct.nsamples)")
    println(io, "")

    # Header
    println(io, "  Parameter    Estimate  Std. Error        MCSE     P-value")
    println(io, "  ---------    --------  ----------        ----     -------")

    # Data row
    for i in 1:length(ct.coef)
        println(
            io, "  ", lpad(ct.coefnames[i], 9), " ",
            to_s(ct.coef[i], 12, 4), " ",
            to_s(ct.stderror[i], 12, 4), " ",
            to_s(ct.mcse[i], 12, 4), " ",
            to_s(ct.pvalue[i], 12, 4)
        )
    end

    println(io, "")
    println(io, "HPD Credible Intervals:")
    for i in 1:length(ct.coef)
        println(io, "  $(ct.coefnames[i]): [$(round(ct.cilower[i], digits = 4)), $(round(ct.ciupper[i], digits = 4))]")
    end

    # Diagnostic information
    println(io, "")
    println(io, "Diagnostics:")
    return if ct.method_type == :MCMC
        println(io, "  Effective Sample Size (ESS): $(round(ct.ess[1], digits = 1))")
    elseif ct.method_type == :VI && ct.elbo !== nothing
        println(io, "  Final ELBO: $(round(ct.elbo, digits = 2))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ct::BDMLCoeftable)
    return show(io, ct)
end

# Pretty printing for BDMLMCMCResult using coeftable
function Base.show(io::IO, r::BDMLMCMCResult)
    # First show the basic info
    println(io, "BDMLMCMCResult ($(r.model_type))")

    # Then show the coeftable
    ct = coeftable(r)
    return show(io, ct)
end

function Base.show(io::IO, ::MIME"text/plain", r::BDMLMCMCResult)
    return show(io, r)
end

# Pretty printing for BDMLVIResult using coeftable
function Base.show(io::IO, r::BDMLVIResult)
    # First show the basic info
    println(io, "BDMLVIResult ($(r.model_type), $(r.variational_family))")

    # Then show the coeftable
    ct = coeftable(r)
    return show(io, ct)
end

function Base.show(io::IO, ::MIME"text/plain", r::BDMLVIResult)
    return show(io, r)
end

# Convenience accessors for BDMLCoeftable
"""
    confint(ct::BDMLCoeftable)

Return confidence/credible intervals as matrix [lower upper].
"""
function confint(ct::BDMLCoeftable)
    return hcat(ct.cilower, ct.ciupper)
end

"""
    ess(ct::BDMLCoeftable)

Return effective sample size.
"""
ess(ct::BDMLCoeftable) = ct.ess

"""
    pvalues(ct::BDMLCoeftable)

Return p-values.
"""
pvalues(ct::BDMLCoeftable) = ct.pvalue

"""
    ess(result::BDMLMCMCResult)

Compute effective sample size from MCMC chain.
Uses MCMCChains.ess for proper multi-chain ESS calculation.
Returns the minimum ESS across all parameters as conservative estimate.
"""
function ess(result::BDMLMCMCResult)
    # Extract ESS from chain using MCMCChains
    ess_df = MCMCChains.ess(result.chain)
    # Get minimum ESS across all parameters (conservative estimate)
    ess_values = ess_df.nt.ess
    finite_ess = filter(x -> isfinite(x) && x > 0, ess_values)
    return length(finite_ess) > 0 ? minimum(finite_ess) : 0.0
end

"""
    rhat(result::BDMLMCMCResult)

Compute R-hat (potential scale reduction factor) from MCMC chain.
R-hat ≈ 1.0 indicates good convergence. Values > 1.05 suggest non-convergence.
Returns the maximum R-hat across all parameters as conservative estimate.
"""
function rhat(result::BDMLMCMCResult)
    # Get Gelman diagnostic (R-hat) from MCMCChains
    gd = gelmandiag(result.chain)
    # Get maximum PSRF (R-hat) across all parameters (conservative estimate)
    rhat_values = gd.nt.psrf
    finite_rhat = filter(x -> isfinite(x) && x > 0, rhat_values)
    return length(finite_rhat) > 0 ? maximum(finite_rhat) : missing
end

"""
    mcse(result::BDMLMCMCResult)

Compute Monte Carlo Standard Error from MCMC chain.
Uses MCMCChains.mcse for proper multi-chain MCSE calculation.
Returns the maximum MCSE across all parameters as conservative estimate.
"""
function mcse(result::BDMLMCMCResult)
    mcse_df = MCMCChains.mcse(result.chain)
    mcse_values = mcse_df.nt.mcse
    finite_mcse = filter(x -> isfinite(x) && x > 0, mcse_values)
    return length(finite_mcse) > 0 ? maximum(finite_mcse) : std(result.alpha_samples) / sqrt(length(result.alpha_samples))
end

"""
    confint(result::AbstractBDMLResult; level=0.95)

Return confidence/credible intervals for the treatment effect from a BDML result.
Computes HPD credible intervals at the specified level.
"""
function confint(result::AbstractBDMLResult; level = 0.95)
    ct = coeftable(result; level = level)
    return hcat(ct.cilower, ct.ciupper)
end

"""
    pvalues(result::AbstractBDMLResult)

Return p-value for testing H0: α = 0 from a BDML result.
"""
function pvalues(result::AbstractBDMLResult)
    ct = coeftable(result)
    return ct.pvalue[1]
end

"""
    rhat_statistic(result::BDMLMCMCResult)

Alias for `rhat(result)` - compute R-hat convergence diagnostic.
R-hat ≈ 1.0 indicates good convergence.
"""
const rhat_statistic = rhat

"""
    chain_info(result::BDMLMCMCResult)

Get chain summary information: (n_chains, n_samples_per_chain, total_samples)
"""
function chain_info(result::BDMLMCMCResult)
    chain = result.chain
    n_chains = size(chain, 3)  # chains are in 3rd dimension
    n_samples_per_chain = size(chain, 1)
    total_samples = n_chains * n_samples_per_chain
    return (n_chains = n_chains, n_samples_per_chain = n_samples_per_chain, total_samples = total_samples)
end

"""
    coef(result::AbstractBDMLResult)

Return coefficient estimates from a BDML result.
"""
function StatsAPI.coef(result::AbstractBDMLResult)
    ct = coeftable(result)
    return ct.coef
end

# Module-level wrapper
function coef(result::AbstractBDMLResult)
    return StatsAPI.coef(result)
end

"""
    stderror(result::AbstractBDMLResult)

Return standard errors from a BDML result.
"""
function StatsAPI.stderror(result::AbstractBDMLResult)
    ct = coeftable(result)
    return ct.stderror
end

# Module-level wrapper
function stderror(result::AbstractBDMLResult)
    return StatsAPI.stderror(result)
end

"""
    vcov(result::AbstractBDMLResult)

Return variance-covariance matrix (diagonal only for single parameter α).
"""
function StatsAPI.vcov(result::AbstractBDMLResult)
    ct = coeftable(result)
    # For single parameter, return diagonal matrix
    return Diagonal(ct.stderror .^ 2)
end

# Module-level wrapper
function vcov(result::AbstractBDMLResult)
    return StatsAPI.vcov(result)
end

# Export additional StatsAPI functions
export coef, stderror, vcov
