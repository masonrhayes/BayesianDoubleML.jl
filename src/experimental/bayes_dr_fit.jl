"""
Fit both Bayes-DR nuisance models and construct posterior-corrected ATE
inference.

The implementation follows the posterior-averaged doubly robust estimator of
Antonelli et al. (2022): Equations 1-2 define the point estimator, the empirical
bootstrap supplies the naive sampling variance in Equation 3, and posterior
variation of the nuisance-dependent estimator supplies the correction in
Equation 4.
"""
struct BayesDRSamplingPlan
    n_samples::Int
    n_burn::Int
    thin::Int
    n_chains::Int
    n_boot::Int
    level::Float64

    function BayesDRSamplingPlan(
            n_samples::Int, n_burn::Int, thin::Int, n_chains::Int,
            n_boot::Int, level::Float64,
        )
        n_samples > 1 || throw(ArgumentError("n_samples must exceed one"))
        n_burn >= 0 || throw(ArgumentError("n_burn must be nonnegative"))
        thin > 0 || throw(ArgumentError("thin must be positive"))
        n_chains > 0 || throw(ArgumentError("n_chains must be positive"))
        n_boot > 1 || throw(ArgumentError("n_boot must exceed one"))
        0 < level < 1 || throw(ArgumentError("level must lie in (0, 1)"))
        return new(n_samples, n_burn, thin, n_chains, n_boot, level)
    end
end

function BayesDRSamplingPlan(
        n_samples::Int, n_burn::Int, thin::Int, n_chains::Int,
        n_boot::Int, level::Real,
    )
    return BayesDRSamplingPlan(
        n_samples, n_burn, thin, n_chains, n_boot, Float64(level),
    )
end

function _sample_nuisance_posteriors(rng, model, method, plan)
    continuous = model.treatment_type === :continuous
    treatment = continuous ?
        (model.T .- model.stats.D_mean) ./ model.stats.D_sd : model.T
    treatment_chains = Vector{BayesDRNuisancePosterior}(undef, plan.n_chains)
    outcome_chains = Vector{BayesDRNuisancePosterior}(undef, plan.n_chains)
    for chain in 1:plan.n_chains
        treatment_chains[chain] = if continuous
            _sample_continuous_treatment_chain(
                rng, treatment, model.X, method, plan.n_samples, plan.n_burn,
                plan.thin, chain,
            )
        else
            _sample_treatment_chain(
                rng, treatment, model.X, method, plan.n_samples, plan.n_burn,
                plan.thin, chain,
            )
        end
        outcome_chains[chain] = _sample_outcome_chain(
            rng, model.Y, treatment, model.X, method, plan.n_samples,
            plan.n_burn, plan.thin, chain; continuous,
        )
    end
    return _combine_posteriors(treatment_chains), _combine_posteriors(outcome_chains)
end

function _fit_binary_bayes_dr(
        model::BayesDRModel, method::BayesDRMCMCMethod,
        plan::BayesDRSamplingPlan, rng::AbstractRNG,
    )
    # Fit treatment and outcome nuisance models independently within each chain.
    treatment_posterior, outcome_posterior =
        _sample_nuisance_posteriors(rng, model, method, plan)

    # Evaluate the doubly robust score for every nuisance-posterior draw.
    propensity, outcome0, outcome1, clipped_fraction = _bayes_dr_predictions(
        model, treatment_posterior, outcome_posterior, method,
    )
    contributions = _aipw_contributions(
        model.Y, model.T, propensity, outcome0, outcome1,
    )
    posterior_effects = vec(mean(contributions; dims = 2))
    estimate = mean(posterior_effects)

    # The score averaged over posterior draws is sufficient for each empirical
    # bootstrap replicate, avoiding a draw-by-observation matrix product.
    bootstrap_estimates = Vector{Float64}(undef, plan.n_boot)
    bootstrap_counts = zeros(Float64, model.n)
    mean_contribution = vec(mean(contributions; dims = 1))
    for bootstrap in 1:plan.n_boot
        fill!(bootstrap_counts, 0)
        for _ in 1:model.n
            bootstrap_counts[rand(rng, 1:model.n)] += 1
        end
        bootstrap_estimates[bootstrap] = dot(mean_contribution, bootstrap_counts) / model.n
    end

    naive_variance = var(bootstrap_estimates)
    posterior_variance = var(posterior_effects)
    # Equation 4 adds nuisance-posterior uncertainty to the naive variance.
    total_variance = naive_variance + posterior_variance
    standard_error = sqrt(total_variance)

    interval = _bayes_dr_interval(
        estimate, posterior_effects, bootstrap_estimates, plan.level,
    )

    return BayesDRResult(
        estimate,
        standard_error,
        interval,
        plan.level,
        naive_variance,
        posterior_variance,
        posterior_effects,
        bootstrap_estimates,
        vec(mean(propensity; dims = 1)),
        clipped_fraction,
        treatment_posterior,
        outcome_posterior,
        (method.propensity_lower, method.propensity_upper),
    )
end

function _curve_covariance(curves)
    centered = curves .- mean(curves; dims = 1)
    return centered' * centered / (size(curves, 1) - 1)
end

function _fit_continuous_bayes_dr(
        model::BayesDRModel, method::BayesDRMCMCMethod,
        plan::BayesDRSamplingPlan, treatment_grid, rng::AbstractRNG,
    )
    length(unique(model.T)) >= method.curve_degree + 1 || throw(
        ArgumentError(
            "continuous T must have at least $(method.curve_degree + 1) distinct values " *
                "for curve_degree=$(method.curve_degree)",
        )
    )

    grid = if treatment_grid === nothing
        lower, upper = quantile(model.T, [0.05, 0.95])
        collect(range(lower, upper; length = 20))
    else
        Vector{Float64}(treatment_grid)
    end
    isempty(grid) && throw(ArgumentError("treatment_grid must not be empty"))
    all(isfinite, grid) || throw(ArgumentError("treatment_grid must contain only finite values"))

    treatment_posterior, outcome_posterior =
        _sample_nuisance_posteriors(rng, model, method, plan)
    design = _continuous_design(model, method, grid)

    posterior_curves, mean_pseudo_outcome, clipped_fraction = _continuous_posterior_curves(
        model, treatment_posterior, outcome_posterior, method, design, grid,
    )
    estimate = vec(mean(posterior_curves; dims = 1))
    bootstrap_curves = _continuous_bootstrap_curves(
        rng, model, design, mean_pseudo_outcome, plan.n_boot,
    )
    naive_covariance = _curve_covariance(bootstrap_curves)
    posterior_covariance = _curve_covariance(posterior_curves)
    total_covariance = naive_covariance + posterior_covariance
    standard_error = sqrt.(max.(diag(total_covariance), 0.0))

    interval = Matrix{Float64}(undef, length(grid), 2)
    for location in eachindex(grid)
        limits = _bayes_dr_interval(
            estimate[location], view(posterior_curves, :, location),
            view(bootstrap_curves, :, location), plan.level,
        )
        interval[location, 1] = limits[1]
        interval[location, 2] = limits[2]
    end

    return BayesDRCurveResult(
        grid,
        estimate,
        standard_error,
        interval,
        plan.level,
        naive_covariance,
        posterior_covariance,
        posterior_curves,
        bootstrap_curves,
        clipped_fraction,
        treatment_posterior,
        outcome_posterior,
        method.curve_degree,
        (method.density_ratio_lower, method.density_ratio_upper),
    )
end

function _fit_impl(
        model::BayesDRModel, method::BayesDRMCMCMethod;
        n_samples::Int = 1000,
        n_burn::Int = 500,
        thin::Int = 1,
        n_chains::Int = 2,
        n_boot::Int = 500,
        level::Real = 0.95,
        treatment_grid = nothing,
        rng::AbstractRNG = Random.default_rng(),
    )
    plan = BayesDRSamplingPlan(n_samples, n_burn, thin, n_chains, n_boot, level)
    return if model.treatment_type === :binary
        treatment_grid === nothing || throw(
            ArgumentError("treatment_grid is only available for continuous treatment"),
        )
        _fit_binary_bayes_dr(model, method, plan, rng)
    else
        _fit_continuous_bayes_dr(model, method, plan, treatment_grid, rng)
    end
end

"""
Construct a posterior-corrected frequentist confidence interval.

The centered Cartesian sum retains the empirical shapes of the bootstrap and
nuisance-posterior uncertainty components whose variances are combined in
Equation 4 of Antonelli et al. (2022). This is not a posterior credible
interval.
"""
function _bayes_dr_interval(estimate, posterior_effects, bootstrap_estimates, level)
    n_boot = length(bootstrap_estimates)
    n_draws = length(posterior_effects)
    interval_draws = Vector{Float64}(undef, n_boot * n_draws)
    bootstrap_center = mean(bootstrap_estimates)
    posterior_center = mean(posterior_effects)
    position = 0
    for bootstrap_estimate in bootstrap_estimates
        data_deviation = bootstrap_estimate - bootstrap_center
        for posterior_effect in posterior_effects
            position += 1
            interval_draws[position] = estimate + data_deviation + posterior_effect - posterior_center
        end
    end
    tail = (1 - level) / 2
    limits = quantile(interval_draws, [tail, 1 - tail])
    return (limits[1], limits[2])
end

function fit!(model::BayesDRModel; force::Bool = false, kwargs...)
    return fit!(model, BayesDRMCMC(); force, kwargs...)
end
