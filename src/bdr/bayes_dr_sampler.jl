function _logaddexp(a::Float64, b::Float64)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function _inclusion_conditional(
        x, x_norm2, residual, sigma2, slab_variance, inclusion_probability,
    )
    precision = x_norm2 + inv(slab_variance)
    posterior_variance = sigma2 / precision
    posterior_mean = dot(x, residual) / precision
    log_p0 = log1p(-inclusion_probability)
    log_p1 = log(inclusion_probability) +
        0.5 * (log(posterior_variance) - log(sigma2 * slab_variance)) +
        0.5 * posterior_mean^2 / posterior_variance
    log_inclusion_probability = log_p1 - _logaddexp(log_p0, log_p1)
    return posterior_mean, posterior_variance, log_inclusion_probability
end

"""
Sample one spike-and-slab coefficient and its inclusion indicator conditional
on the current partial residual. `x_norm2` is cached once per chain because the
standardized design matrix does not change across Gibbs iterations.
"""
function _sample_inclusion_coefficient(
        rng, x, x_norm2, residual, sigma2, slab_variance, inclusion_probability,
    )
    posterior_mean, posterior_variance, log_inclusion_probability =
        _inclusion_conditional(
        x, x_norm2, residual, sigma2, slab_variance, inclusion_probability,
    )
    included = log(rand(rng)) < log_inclusion_probability
    coefficient = included ? posterior_mean + sqrt(posterior_variance) * randn(rng) : 0.0
    return coefficient, included
end

function _sample_sparse_coefficients!(
        rng, coefficients, inclusion, residual, X, covariate_norm2,
        sigma2, slab_variance, inclusion_probability,
    )
    for j in axes(X, 2)
        xj = view(X, :, j)
        residual .+= xj .* coefficients[j]
        coefficients[j], inclusion[j] = _sample_inclusion_coefficient(
            rng, xj, covariate_norm2[j], residual, sigma2, slab_variance,
            inclusion_probability,
        )
        residual .-= xj .* coefficients[j]
    end
    return count(inclusion)
end

function _sample_sparsity_hyperparameters(
        rng, coefficients, n_included, sigma2, method,
    )
    coefficient_norm2 = dot(coefficients, coefficients)
    slab_variance = rand(
        rng,
        InverseGamma(
            method.slab_shape + n_included / 2,
            method.slab_scale + coefficient_norm2 / (2 * sigma2),
        ),
    )
    inclusion_probability = rand(
        rng,
        Beta(
            method.inclusion_a + n_included,
            method.inclusion_b_scale * length(coefficients) +
                length(coefficients) - n_included,
        ),
    )
    return slab_variance, inclusion_probability
end

"""
Update coefficients that are always included in a Gaussian linear model.

The design factorization and work vectors are supplied by the caller so the
Gibbs loop can perform the conjugate normal update without repeated
factorizations or temporary allocations.
"""
function _sample_unpenalized!(
        rng, coefficients, design, factor, response, nuisance, sigma2, target, noise,
    )
    @. target = response - nuisance
    mul!(coefficients, transpose(design), target)
    ldiv!(factor, coefficients)
    randn!(rng, noise)
    ldiv!(factor.U, noise)
    @. coefficients += sqrt(sigma2) * noise
    return coefficients
end

"""Sample a Gaussian nuisance model with an always-included design matrix."""
function _sample_gaussian_chain(
        rng, response, unpenalized, X, method, n_samples, n_burn, thin, chain_id,
    )
    n, p = size(X)
    n_unpenalized = size(unpenalized, 2)
    total_iterations = n_burn + n_samples * thin
    unpenalized_factor = cholesky(Symmetric(unpenalized' * unpenalized))
    unpenalized_coefficients = zeros(n_unpenalized)
    covariate_coefficients = zeros(p)
    inclusion_indicators = falses(p)
    covariate_norm2 = vec(sum(abs2, X; dims = 1))
    covariate_predictor = zeros(n)
    unpenalized_predictor = zeros(n)
    residual = zeros(n)
    unpenalized_noise = zeros(n_unpenalized)
    sigma2 = max(var(response), 0.1)
    slab_variance = 1.0
    inclusion_probability = 0.1

    coefficients = Matrix{Float64}(undef, n_samples, p + n_unpenalized)
    inclusion = falses(n_samples, p)
    residual_variance = Vector{Float64}(undef, n_samples)
    slab_variances = Vector{Float64}(undef, n_samples)
    inclusion_probabilities = Vector{Float64}(undef, n_samples)
    keep = 0

    for iteration in 1:total_iterations
        mul!(covariate_predictor, X, covariate_coefficients)
        _sample_unpenalized!(
            rng, unpenalized_coefficients, unpenalized, unpenalized_factor, response,
            covariate_predictor, sigma2, residual, unpenalized_noise,
        )
        mul!(unpenalized_predictor, unpenalized, unpenalized_coefficients)
        @. residual = response - unpenalized_predictor - covariate_predictor
        n_included = _sample_sparse_coefficients!(
            rng, covariate_coefficients, inclusion_indicators, residual, X,
            covariate_norm2, sigma2, slab_variance, inclusion_probability,
        )
        sigma_shape = method.sigma_shape + (n + n_included) / 2
        sigma_scale = method.sigma_scale + dot(residual, residual) / 2 +
            dot(covariate_coefficients, covariate_coefficients) / (2 * slab_variance)
        sigma2 = rand(rng, InverseGamma(sigma_shape, sigma_scale))
        slab_variance, inclusion_probability = _sample_sparsity_hyperparameters(
            rng, covariate_coefficients, n_included, sigma2, method,
        )

        if iteration > n_burn && (iteration - n_burn) % thin == 0
            keep += 1
            @views coefficients[keep, 1:n_unpenalized] .= unpenalized_coefficients
            @views coefficients[keep, (n_unpenalized + 1):end] .= covariate_coefficients
            inclusion[keep, :] .= inclusion_indicators
            residual_variance[keep] = sigma2
            slab_variances[keep] = slab_variance
            inclusion_probabilities[keep] = inclusion_probability
        end
    end
    chain_ids = fill(chain_id, n_samples)
    return BayesDRNuisancePosterior(
        coefficients, inclusion, residual_variance, slab_variances,
        inclusion_probabilities, chain_ids,
    )
end

function _treatment_basis(T, degree)
    basis = Matrix{Float64}(undef, length(T), degree)
    @views basis[:, 1] .= T
    for power in 2:degree
        @views basis[:, power] .= basis[:, power - 1] .* T
    end
    return basis
end

function _sample_outcome_chain(
        rng, Y, T, X, method, n_samples, n_burn, thin, chain_id;
        continuous::Bool = false,
    )
    treatment_design = continuous ? _treatment_basis(T, method.curve_degree) : reshape(T, :, 1)
    unpenalized = hcat(ones(length(T)), treatment_design)
    return _sample_gaussian_chain(
        rng, Y, unpenalized, X, method, n_samples, n_burn, thin, chain_id,
    )
end

function _sample_continuous_treatment_chain(
        rng, T, X, method, n_samples, n_burn, thin, chain_id,
    )
    return _sample_gaussian_chain(
        rng, T, ones(length(T), 1), X, method, n_samples, n_burn, thin, chain_id,
    )
end

"""
Sample latent Gaussian variables for the probit treatment model, truncated
above or below zero according to observed treatment.
"""
function _sample_latent_probit!(rng, latent, T, linear_predictor)
    @inbounds for i in eachindex(T)
        distribution = if T[i] == 1
            truncated(Normal(linear_predictor[i], 1), 0, Inf)
        else
            truncated(Normal(linear_predictor[i], 1), -Inf, 0)
        end
        latent[i] = rand(rng, distribution)
    end
    return latent
end

"""
Sample the probit treatment nuisance model for one chain.

The intercept is always included. Covariate coefficients receive independent
spike-and-slab updates after sampling the latent Gaussian treatment response.
This is the binary-treatment nuisance model used by the linear Bayes-DR
implementation of Antonelli et al. (2022).
"""
function _sample_treatment_chain(rng, T, X, method, n_samples, n_burn, thin, chain_id)
    n, p = size(X)
    total_iterations = n_burn + n_samples * thin
    intercept = 0.0
    covariate_coefficients = zeros(p)
    inclusion_indicators = falses(p)
    latent = similar(T)
    covariate_norm2 = vec(sum(abs2, X; dims = 1))
    linear_predictor = zeros(n)
    covariate_predictor = zeros(n)
    residual = zeros(n)
    slab_variance = 1.0
    inclusion_probability = 0.1

    coefficients = Matrix{Float64}(undef, n_samples, p + 1)
    inclusion = falses(n_samples, p)
    slab_variances = Vector{Float64}(undef, n_samples)
    inclusion_probabilities = Vector{Float64}(undef, n_samples)
    keep = 0

    for iteration in 1:total_iterations
        mul!(covariate_predictor, X, covariate_coefficients)
        @. linear_predictor = intercept + covariate_predictor
        _sample_latent_probit!(rng, latent, T, linear_predictor)
        intercept = (sum(latent) - sum(covariate_predictor)) / n + randn(rng) / sqrt(n)
        @. residual = latent - intercept - covariate_predictor
        n_included = _sample_sparse_coefficients!(
            rng, covariate_coefficients, inclusion_indicators, residual, X,
            covariate_norm2, 1.0, slab_variance, inclusion_probability,
        )
        slab_variance, inclusion_probability = _sample_sparsity_hyperparameters(
            rng, covariate_coefficients, n_included, 1.0, method,
        )

        if iteration > n_burn && (iteration - n_burn) % thin == 0
            keep += 1
            coefficients[keep, 1] = intercept
            @views coefficients[keep, 2:end] .= covariate_coefficients
            inclusion[keep, :] .= inclusion_indicators
            slab_variances[keep] = slab_variance
            inclusion_probabilities[keep] = inclusion_probability
        end
    end
    return BayesDRNuisancePosterior(
        coefficients, inclusion, ones(n_samples), slab_variances,
        inclusion_probabilities, fill(chain_id, n_samples),
    )
end

function _combine_posteriors(posteriors::Vector{BayesDRNuisancePosterior})
    return BayesDRNuisancePosterior(
        reduce(vcat, getfield.(posteriors, :coefficients)),
        reduce(vcat, getfield.(posteriors, :inclusion)),
        reduce(vcat, getfield.(posteriors, :residual_variance)),
        reduce(vcat, getfield.(posteriors, :slab_variance)),
        reduce(vcat, getfield.(posteriors, :inclusion_probability)),
        reduce(vcat, getfield.(posteriors, :chain_id)),
    )
end

function _posterior_chain_array(posterior::BayesDRNuisancePosterior)
    chain_ids = sort!(unique(posterior.chain_id))
    isempty(chain_ids) && throw(ArgumentError("nuisance posterior contains no chains"))
    rows_by_chain = [findall(==(chain), posterior.chain_id) for chain in chain_ids]
    n_iterations = length(first(rows_by_chain))
    all(rows -> length(rows) == n_iterations, rows_by_chain) || throw(
        ArgumentError("all nuisance-posterior chains must contain the same number of draws"),
    )

    n_coefficients = size(posterior.coefficients, 2)
    n_inclusion = size(posterior.inclusion, 2)
    n_parameters = n_coefficients + n_inclusion + 3
    samples = Array{Float64}(undef, n_iterations, length(chain_ids), n_parameters)
    for (chain_index, rows) in pairs(rows_by_chain)
        coefficient_range = 1:n_coefficients
        inclusion_range = (n_coefficients + 1):(n_coefficients + n_inclusion)
        samples[:, chain_index, coefficient_range] .= posterior.coefficients[rows, :]
        samples[:, chain_index, inclusion_range] .= posterior.inclusion[rows, :]
        samples[:, chain_index, end - 2] .= posterior.residual_variance[rows]
        samples[:, chain_index, end - 1] .= posterior.slab_variance[rows]
        samples[:, chain_index, end] .= posterior.inclusion_probability[rows]
    end
    return samples
end

function _normal_cdf(x)
    return cdf(Normal(), x)
end

"""
Evaluate nuisance predictions for every retained posterior draw.

Returns clipped propensity scores, potential-outcome predictions under
``T=0`` and ``T=1``, and the fraction of propensity predictions outside the
requested bounds before clipping. The additive outcome model makes the
treatment coefficient the draw-specific conditional treatment effect.
"""
function _bayes_dr_predictions(model, treatment_posterior, outcome_posterior, method)
    n_draws = size(treatment_posterior.coefficients, 1)
    n = model.n

    propensity = @views treatment_posterior.coefficients[:, 2:end] * model.X'
    propensity .+= view(treatment_posterior.coefficients, :, 1)
    propensity .= _normal_cdf.(propensity)
    lower = max(method.propensity_lower, eps(Float64))
    upper = min(method.propensity_upper, 1 - eps(Float64))
    clipped_fraction = count(p -> p < lower || p > upper, propensity) / length(propensity)
    clamp!(propensity, lower, upper)
    outcome0 = @views outcome_posterior.coefficients[:, 3:end] * model.X'
    outcome0 .+= view(outcome_posterior.coefficients, :, 1)
    outcome1 = outcome0 .+ view(outcome_posterior.coefficients, :, 2)
    size(propensity) == (n_draws, n) || error("invalid propensity prediction dimensions")
    return propensity, outcome0, outcome1, clipped_fraction
end

"""
Compute observation-level augmented inverse-probability weighted contributions.

For posterior draw ``b`` and observation ``i``, this evaluates the doubly
robust score in Equation 1 of Antonelli et al. (2022). Averaging across
observations gives the draw-specific ATE estimate; averaging those estimates
across nuisance-posterior draws gives the Bayes-DR point estimate in Equation
2.
"""
function _aipw_contributions(Y, T, propensity, outcome0, outcome1)
    n_draws, n = size(propensity)
    contributions = Matrix{Float64}(undef, n_draws, n)
    @inbounds for i in 1:n, b in 1:n_draws
        p = propensity[b, i]
        contributions[b, i] = outcome1[b, i] - outcome0[b, i] +
            T[i] * (Y[i] - outcome1[b, i]) / p -
            (1 - T[i]) * (Y[i] - outcome0[b, i]) / (1 - p)
    end
    return contributions
end

function _continuous_design(model, method, treatment_grid)
    treatment = (model.T .- model.stats.D_mean) ./ model.stats.D_sd
    treatment_basis = _treatment_basis(treatment, method.curve_degree)
    regression_design = hcat(ones(model.n), treatment_basis)
    standardized_grid = (treatment_grid .- model.stats.D_mean) ./ model.stats.D_sd
    grid_design = hcat(
        ones(length(treatment_grid)),
        _treatment_basis(standardized_grid, method.curve_degree),
    )
    return (
        treatment,
        treatment_basis,
        regression_design,
        regression_factor = qr(regression_design, ColumnNorm()),
        grid_design,
    )
end

function _continuous_curve_for_draw(
        model, treatment_posterior, outcome_posterior, method, draw, design,
    )
    degree = method.curve_degree
    n = model.n
    treatment = design.treatment
    treatment_basis = design.treatment_basis
    covariates = model.X

    treatment_coefficients = view(treatment_posterior.coefficients, draw, :)
    treatment_mean = covariates * view(treatment_coefficients, 2:length(treatment_coefficients))
    treatment_mean .+= treatment_coefficients[1]
    treatment_variance = treatment_posterior.residual_variance[draw]
    log_density_constant = -0.5 * log(2π * treatment_variance)
    inverse_twice_variance = inv(2 * treatment_variance)

    outcome_coefficients = view(outcome_posterior.coefficients, draw, :)
    covariate_start = degree + 2
    covariate_predictor = covariates * view(outcome_coefficients, covariate_start:length(outcome_coefficients))
    outcome_mean = treatment_basis * view(outcome_coefficients, 2:(degree + 1))
    outcome_mean .+= outcome_coefficients[1]
    outcome_mean .+= covariate_predictor
    marginal_covariate_mean = mean(covariate_predictor)

    pseudo_outcome = Vector{Float64}(undef, n)
    clipped = 0
    log_lower = log(method.density_ratio_lower)
    log_upper = log(method.density_ratio_upper)
    @inbounds for i in 1:n
        log_marginal_density = -Inf
        for j in 1:n
            difference = treatment[i] - treatment_mean[j]
            log_marginal_density = _logaddexp(
                log_marginal_density,
                log_density_constant - difference^2 * inverse_twice_variance,
            )
        end
        log_marginal_density -= log(n)
        difference = treatment[i] - treatment_mean[i]
        log_conditional_density =
            log_density_constant - difference^2 * inverse_twice_variance
        log_ratio = log_marginal_density - log_conditional_density
        clipped += log_ratio < log_lower || log_ratio > log_upper
        ratio = exp(clamp(log_ratio, log_lower, log_upper))
        marginal_outcome = outcome_coefficients[1] +
            dot(view(treatment_basis, i, :), view(outcome_coefficients, 2:(degree + 1))) +
            marginal_covariate_mean
        pseudo_outcome[i] = (model.Y[i] - outcome_mean[i]) * ratio + marginal_outcome
    end

    curve_coefficients = design.regression_factor \ pseudo_outcome
    return design.grid_design * curve_coefficients, pseudo_outcome, clipped
end

function _continuous_posterior_curves(
        model, treatment_posterior, outcome_posterior, method, design, treatment_grid,
    )
    n_draws = size(treatment_posterior.coefficients, 1)
    curves = Matrix{Float64}(undef, n_draws, length(treatment_grid))
    mean_pseudo_outcome = zeros(model.n)
    clipped = 0
    for draw in 1:n_draws
        curve, pseudo_outcome, draw_clipped = _continuous_curve_for_draw(
            model, treatment_posterior, outcome_posterior, method, draw, design,
        )
        curves[draw, :] .= curve
        mean_pseudo_outcome .+= pseudo_outcome
        clipped += draw_clipped
    end
    mean_pseudo_outcome ./= n_draws
    return curves, mean_pseudo_outcome, clipped / (n_draws * model.n)
end

function _continuous_bootstrap_curves(
        rng, model, design, mean_pseudo_outcome, n_boot,
    )
    # As in the binary path, hold posterior-averaged nuisance quantities fixed
    # and bootstrap the observation-level estimating data.
    curves = Matrix{Float64}(undef, n_boot, size(design.grid_design, 1))
    indices = Vector{Int}(undef, model.n)
    for bootstrap in 1:n_boot
        rand!(rng, indices, 1:model.n)
        coefficients = qr(view(design.regression_design, indices, :), ColumnNorm()) \
            view(mean_pseudo_outcome, indices)
        curves[bootstrap, :] .= design.grid_design * coefficients
    end
    return curves
end
