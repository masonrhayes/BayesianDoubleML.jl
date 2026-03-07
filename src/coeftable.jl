# Coeftable implementation following StatsAPI conventions
# Provides statistical summaries for BDML results with HPD intervals

using StatsAPI

# Helper function for formatting numbers with fixed width
to_s(x, width = 12, digits = 4) = lpad(round(x, digits = digits), width)

"""
    hpd_interval(samples::Vector{Float64}, level::Real=0.95)

Compute the Highest Posterior Density (HPD) interval for a vector of samples.
HPD intervals are the shortest intervals containing the specified probability mass.
"""
function hpd_interval(samples::Vector{Float64}, level::Real = 0.95)
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
    mcse(samples::Vector{Float64})

Compute the Monte Carlo Standard Error using batch means method.
This estimates the error due to finite sampling.
"""
function mcse(samples::Vector{Float64})
    n = length(samples)

    # If samples are too few, use simple std/sqrt(n)
    if n < 100
        return std(samples) / sqrt(n)
    end

    # Batch means method
    n_batches = max(2, floor(Int, sqrt(n)))
    batch_size = div(n, n_batches)

    batch_means = Float64[]
    for i in 1:n_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, n)
        push!(batch_means, mean(samples[start_idx:end_idx]))
    end

    return std(batch_means) / sqrt(n_batches)
end

"""
    effective_sample_size(samples::Vector{Float64})

Compute effective sample size using autocorrelation method.
ESS accounts for correlation in MCMC samples.
"""
function effective_sample_size(samples::Vector{Float64})
    n = length(samples)

    # If using MCMCChains, extract from chain
    if n < 10
        return Float64(n)
    end

    # Compute autocorrelation
    mean_samples = mean(samples)
    var_samples = var(samples)

    if var_samples < 1.0e-10
        return Float64(n)
    end

    # Compute autocorrelations up to lag where they become negligible
    max_lag = min(n - 1, 100)
    autocorr_sum = 0.0

    for lag in 1:max_lag
        # Compute autocorrelation at this lag
        c = 0.0
        for i in 1:(n - lag)
            c += (samples[i] - mean_samples) * (samples[i + lag] - mean_samples)
        end
        c = c / ((n - lag) * var_samples)

        # Stop when autocorrelation becomes negative (Geyer's method)
        if c < 0
            break
        end

        autocorr_sum += c
    end

    ess = n / (1 + 2 * autocorr_sum)
    return max(1.0, ess)
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
    coeftable(result::BDMLResult; level=0.95)

Compute coefficient table for MCMC results with HPD credible intervals.
"""
function coeftable(result::BDMLResult; level = 0.95)
    samples = result.alpha_samples

    # Basic statistics
    coef_est = mean(samples)
    std_error = std(samples)
    mcse_val = mcse(samples)

    # HPD interval
    hpd = hpd_interval(samples, level)

    # P-value
    pval = compute_pvalue(samples)

    # ESS
    ess_val = effective_sample_size(samples)

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
"""
function coeftable(result::BDMLVIResult; level = 0.95)
    samples = result.alpha_samples

    # Basic statistics
    coef_est = mean(samples)
    std_error = std(samples)
    mcse_val = mcse(samples)

    # HPD interval
    hpd = hpd_interval(samples, level)

    # P-value
    pval = compute_pvalue(samples)

    # ESS for VI (tends to be higher due to independent samples)
    ess_val = effective_sample_size(samples)

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
    if result isa BDMLResult
        return coeftable(result::BDMLResult; level = level)
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
        println(io, "  Effective Sample Size (ESS): $(round(ct.ess[1], digits = 1))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ct::BDMLCoeftable)
    return show(io, ct)
end

# Pretty printing for BDMLResult using coeftable
function Base.show(io::IO, r::BDMLResult)
    # First show the basic info
    println(io, "BDMLResult ($(r.model_type))")

    # Then show the coeftable
    ct = coeftable(r)
    return show(io, ct)
end

function Base.show(io::IO, ::MIME"text/plain", r::BDMLResult)
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
    ess(result::BDMLResult)

Compute effective sample size from MCMC chain.
Uses MCMCChains.ess for proper multi-chain ESS calculation.
Returns the minimum ESS across all parameters as conservative estimate.
"""
function ess(result::BDMLResult)
    # Extract ESS from chain using MCMCChains
    ess_df = MCMCChains.ess(result.chain)
    # Get minimum ESS across all parameters (conservative estimate)
    ess_values = ess_df.nt.ess
    finite_ess = filter(x -> isfinite(x) && x > 0, ess_values)
    return length(finite_ess) > 0 ? minimum(finite_ess) : 0.0
end

"""
    rhat(result::BDMLResult)

Compute R-hat (potential scale reduction factor) from MCMC chain.
R-hat ≈ 1.0 indicates good convergence. Values > 1.05 suggest non-convergence.
Returns the maximum R-hat across all parameters as conservative estimate.
"""
function rhat(result::BDMLResult)
    # Get Gelman diagnostic (R-hat) from MCMCChains
    gd = gelmandiag(result.chain)
    # Get maximum PSRF (R-hat) across all parameters (conservative estimate)
    rhat_values = gd.nt.psrf
    finite_rhat = filter(x -> isfinite(x) && x > 0, rhat_values)
    return length(finite_rhat) > 0 ? maximum(finite_rhat) : missing
end

"""
    mcse(result::BDMLResult)

Compute Monte Carlo Standard Error from MCMC samples.
"""
function mcse(result::BDMLResult)
    return mcse(result.alpha_samples)
end

"""
    chain_info(result::BDMLResult)

Get chain summary information: (n_chains, n_samples_per_chain, total_samples)
"""
function chain_info(result::BDMLResult)
    chain = result.chain
    n_chains = size(chain, 3)  # chains are in 3rd dimension
    n_samples_per_chain = size(chain, 1)
    total_samples = n_chains * n_samples_per_chain
    return (n_chains = n_chains, n_samples_per_chain = n_samples_per_chain, total_samples = total_samples)
end

# Export the coeftable function and related types
export coeftable, BDMLCoeftable, confint, ess, pvalues, hpd_interval, mcse, effective_sample_size, rhat, chain_info
