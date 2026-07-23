# Coeftable implementation following StatsAPI conventions
# Provides statistical summaries for BDML results with HPD intervals

using StatsAPI
using Printf

# Helper function for formatting numbers with fixed width
to_s(x, width = 12, digits = 4) = lpad(round(x, digits = digits), width)

# Helper function to avoid closures in filter operations
isfinite_and_positive(x) = isfinite(x) && x > 0

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

"""
    BDMLCoeftable

Coefficient table following StatsAPI conventions for regression results.

Stores parameter estimates, standard errors, credible intervals, and diagnostics
for Bayesian Double Machine Learning models.

# Fields
- `coefnames::Vector{String}`: Parameter names
- `coef::Vector{Float64}`: Point estimates
- `stderror::Vector{Float64}`: Standard errors
- `mcse::Vector{Float64}`: Monte Carlo standard errors
- `cilower::Vector{Float64}`: Lower bound of credible interval
- `ciupper::Vector{Float64}`: Upper bound of credible interval
- `pvalue::Vector{Float64}`: Two-sided p-values
- `ess::Vector{Float64}`: Effective sample sizes (MCMC only)
- `elbo::Union{Float64, Nothing}`: Final ELBO value (VI only)
- `level::Float64`: Credible interval level (e.g., 0.95)
- `nsamples::Int`: Number of posterior samples
- `model_type::Symbol`: :basic or :hier
- `method_type::Symbol`: :mcmc or :vi

# Usage
```julia
result = fit(problem, MCMCNUTS())
ct = coeftable(result)
println(ct)  # Pretty-printed table
```

See also: [`coeftable`](@ref), [`coef`](@ref), [`stderror`](@ref)
"""
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
Uses FlexiChains for ESS and MCSE calculations.
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

    # ESS from chain (get minimum across all parameters as conservative estimate)
    ess_summary = FlexiChains.ess(result.chain)
    ess_values = Array(ess_summary)
    finite_ess = filter(isfinite_and_positive, ess_values)
    ess_val = length(finite_ess) > 0 ? minimum(finite_ess) : 0.0

    # MCSE from chain (get maximum across all parameters as conservative estimate)
    mcse_summary = FlexiChains.mcse(result.chain)
    mcse_values = Array(mcse_summary)
    finite_mcse = filter(isfinite_and_positive, mcse_values)
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

    # Header - using Printf for consistent alignment
    Printf.@printf(
        io, "  %-9s %12s %12s %12s %12s\n",
        "Parameter", "Estimate", "Std. Error", "MCSE", "P-value"
    )
    Printf.@printf(
        io, "  %-9s %12s %12s %12s %12s\n",
        "---------", "--------", "----------", "----", "-------"
    )

    # Data rows - using Printf for consistent alignment
    for i in 1:length(ct.coef)
        Printf.@printf(
            io, "  %-9s %12.4f %12.4f %12.4f %12.4f\n",
            ct.coefnames[i], ct.coef[i], ct.stderror[i],
            ct.mcse[i], ct.pvalue[i]
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
    # First show the basic info with method type
    method_name = r.vi_method == :simple ? "SimpleVI" : "UnifiedVI"
    println(io, "BDMLVIResult ($(r.model_type), $(method_name), $(r.variational_family))")

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
Uses FlexiChains for proper multi-chain ESS calculation.
Returns the minimum ESS across all parameters as conservative estimate.
"""
function ess(result::BDMLMCMCResult)
    # Extract ESS from chain using FlexiChains
    ess_summary = FlexiChains.ess(result.chain)
    # Get minimum ESS across all parameters (conservative estimate)
    ess_values = Array(ess_summary)
    finite_ess = filter(isfinite_and_positive, ess_values)
    return length(finite_ess) > 0 ? minimum(finite_ess) : 0.0
end

"""
    rhat(result::BDMLMCMCResult)

Compute R-hat (potential scale reduction factor) from MCMC chain.
R-hat ≈ 1.0 indicates good convergence. Values > 1.05 suggest non-convergence.
Returns the maximum R-hat across all parameters as conservative estimate.
"""
function rhat(result::BDMLMCMCResult)
    # Get R-hat diagnostic from FlexiChains
    rhat_summary = FlexiChains.rhat(result.chain)
    # Get maximum R-hat across all parameters (conservative estimate)
    rhat_values = Array(rhat_summary)
    finite_rhat = filter(isfinite_and_positive, rhat_values)
    return length(finite_rhat) > 0 ? maximum(finite_rhat) : missing
end

"""
    mcse(result::BDMLMCMCResult)

Compute Monte Carlo Standard Error from MCMC chain.
Uses FlexiChains for proper multi-chain MCSE calculation.
Returns the maximum MCSE across all parameters as conservative estimate.
"""
function mcse(result::BDMLMCMCResult)
    mcse_summary = FlexiChains.mcse(result.chain)
    mcse_values = Array(mcse_summary)
    finite_mcse = filter(isfinite_and_positive, mcse_values)
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
function rhat_statistic(result::BDMLMCMCResult)
    return rhat(result)
end

"""
    chain_info(result::BDMLMCMCResult)

Get chain summary information: (n_chains, n_samples_per_chain, total_samples)
"""
function chain_info(result::BDMLMCMCResult)
    chain = result.chain
    n_samples_per_chain, n_chains = size(chain[:σ_U])
    total_samples = n_chains * n_samples_per_chain
    return (n_chains = n_chains, n_samples_per_chain = n_samples_per_chain, total_samples = total_samples)
end

"""
    coef(result::AbstractBDMLResult)

Return coefficient estimates from a BDML result.

This function extracts the point estimates for all parameters from the fitted model.
For BDML models, this primarily returns the causal effect α.

# Arguments
- `result::AbstractBDMLResult`: A fitted BDML result (MCMC or VI)

# Returns
- `Vector{Float64}`: Coefficient estimates

# Examples
```julia
result = fit(problem, MCMCNUTS())
estimates = coef(result)
```

See also: [`stderror`](@ref), [`vcov`](@ref), [`coeftable`](@ref)
"""
function StatsAPI.coef(result::AbstractBDMLResult)
    ct = coeftable(result)
    return ct.coef
end

# Module-level wrapper with docstring
"""
    coef(result::AbstractBDMLResult)

Module-level wrapper for `StatsAPI.coef`. See `StatsAPI.coef` for details.
"""
function coef(result::AbstractBDMLResult)
    return StatsAPI.coef(result)
end

"""
    stderror(result::AbstractBDMLResult)

Return standard errors from a BDML result.

Computes the standard errors for all parameter estimates. For MCMC results,
this is based on the posterior standard deviation. For VI results, this is
based on the variational posterior approximation.

# Arguments
- `result::AbstractBDMLResult`: A fitted BDML result (MCMC or VI)

# Returns
- `Vector{Float64}`: Standard errors for each parameter

# Examples
```julia
result = fit(problem, MCMCNUTS())
se = stderror(result)
```

See also: [`coef`](@ref), [`vcov`](@ref), [`coeftable`](@ref)
"""
function StatsAPI.stderror(result::AbstractBDMLResult)
    ct = coeftable(result)
    return ct.stderror
end

# Module-level wrapper with docstring
"""
    stderror(result::AbstractBDMLResult)

Module-level wrapper for `StatsAPI.stderror`. See `StatsAPI.stderror` for details.
"""
function stderror(result::AbstractBDMLResult)
    return StatsAPI.stderror(result)
end

"""
    vcov(result::AbstractBDMLResult)

Return variance-covariance matrix from a BDML result.

For BDML models with a single causal effect parameter α, returns a 1×1 diagonal
matrix containing the variance. For models with multiple parameters, returns
the full variance-covariance matrix.

# Arguments
- `result::AbstractBDMLResult`: A fitted BDML result (MCMC or VI)

# Returns
- `Diagonal{Float64}`: Variance-covariance matrix

# Examples
```julia
result = fit(problem, MCMCNUTS())
varcov = vcov(result)
```

See also: [`coef`](@ref), [`stderror`](@ref), [`coeftable`](@ref)
"""
function StatsAPI.vcov(result::AbstractBDMLResult)
    ct = coeftable(result)
    # For single parameter, return diagonal matrix
    return Diagonal(ct.stderror .^ 2)
end

# Module-level wrapper with docstring
"""
    vcov(result::AbstractBDMLResult)

Module-level wrapper for `StatsAPI.vcov`. See `StatsAPI.vcov` for details.
"""
function vcov(result::AbstractBDMLResult)
    return StatsAPI.vcov(result)
end

# Model delegation - allow calling result extraction functions directly on fitted models

"""
    coeftable(model::AbstractBDMLModel; level=0.95)

Compute coefficient table for a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function coeftable(model::AbstractBDMLModel; level = 0.95)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    return coeftable(model.result; level = level)
end

"""
    extract_alpha(model::AbstractBDMLModel)

Extract alpha samples from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function extract_alpha(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult  # Type assertion after is_fitted check
    return extract_alpha(result)
end

"""
    coef(model::AbstractBDMLModel)

Return coefficient estimates from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function StatsAPI.coef(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return StatsAPI.coef(result)
end

function coef(model::AbstractBDMLModel)
    return StatsAPI.coef(model)
end

"""
    stderror(model::AbstractBDMLModel)

Return standard errors from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function StatsAPI.stderror(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return StatsAPI.stderror(result)
end

function stderror(model::AbstractBDMLModel)
    return StatsAPI.stderror(model)
end

"""
    vcov(model::AbstractBDMLModel)

Return variance-covariance matrix from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function StatsAPI.vcov(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return StatsAPI.vcov(result)
end

function vcov(model::AbstractBDMLModel)
    return StatsAPI.vcov(model)
end

"""
    confint(model::AbstractBDMLModel; level=0.95)

Return confidence/credible intervals from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function confint(model::AbstractBDMLModel; level = 0.95)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return confint(result; level = level)
end

"""
    credible_interval(model::AbstractBDMLModel; level=0.95)

Return credible intervals from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function credible_interval(model::AbstractBDMLModel; level = 0.95)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return credible_interval(extract_alpha(result); level = level)
end

"""
    pvalues(model::AbstractBDMLModel)

Return p-values from a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.
"""
function pvalues(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    return pvalues(result)
end

# MCMC-specific functions
"""
    ess(model::AbstractBDMLModel)

Return effective sample size from a fitted BDML model (MCMC only).

Delegates to the stored result. Throws an error if the model has not been fitted
or if the model was fitted with VI.
"""
function ess(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    result isa BDMLMCMCResult || error("ESS only available for MCMC results.")
    return ess(result)
end

"""
    mcse(model::AbstractBDMLModel)

Return Monte Carlo standard error from a fitted BDML model (MCMC only).

Delegates to the stored result. Throws an error if the model has not been fitted
or if the model was fitted with VI.
"""
function mcse(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    result isa BDMLMCMCResult || error("MCSE only available for MCMC results.")
    return mcse(result)
end

"""
    rhat(model::AbstractBDMLModel)

Return R-hat convergence diagnostic from a fitted BDML model (MCMC only).

Delegates to the stored result. Throws an error if the model has not been fitted
or if the model was fitted with VI.
"""
function rhat(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    result isa BDMLMCMCResult || error("R-hat only available for MCMC results.")
    return rhat(result)
end

"""
    rhat_statistic(model::AbstractBDMLModel)

Alias for `rhat(model)` - compute R-hat convergence diagnostic (MCMC only).
"""
function rhat_statistic(model::AbstractBDMLModel)
    return rhat(model)
end

"""
    chain_info(model::AbstractBDMLModel)

Return chain summary information from a fitted BDML model (MCMC only).

Delegates to the stored result. Throws an error if the model has not been fitted
or if the model was fitted with VI.
"""
function chain_info(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    result = model.result::AbstractBDMLResult
    result isa BDMLMCMCResult || error("Chain info only available for MCMC results.")
    return chain_info(result)
end

# Export additional StatsAPI functions
export coef, stderror, vcov
