"""
    BayesDRModel(Y, T, X; treatment_type=:auto)
    BayesDRModel(df::DataFrame, y::Symbol, treatment::Symbol;
                 x_cols=nothing, treatment_type=:auto)

Experimental Bayes-DR model for binary-treatment ATEs and continuous-treatment
exposure-response curves, based on Antonelli, Papadogeorgou, and Dominici
(2022).

For a binary treatment, the nuisance models are a probit treatment model and a
Gaussian linear outcome model,

```math
P(T_i = 1 \\mid X_i) = \\Phi(\\gamma_0 + X_i'\\gamma),
```

```math
Y_i = \\beta_0 + \\Delta T_i + X_i'\\beta + \\epsilon_i,
\\qquad \\epsilon_i \\sim N(0, \\sigma^2).
```

For a continuous treatment, the treatment nuisance model is Gaussian and the
outcome model includes a polynomial in treatment. The resulting pseudo-outcome
is regressed on treatment to estimate the exposure-response curve.

Spike-and-slab priors perform variable selection separately in the treatment
and outcome models. Covariates are standardized internally. Continuous
treatments are standardized for numerical calculations, while reported grid
locations remain on the original treatment scale.

The `DataFrame` constructor uses every column other than `y` and `treatment` as
a covariate unless `x_cols` is supplied.

# Limitations

This implementation supports continuous outcomes, additive linear nuisance
models, the population ATE for binary treatments, and exposure-response curves
for continuous treatments. It does not currently estimate the ATTE or
heterogeneous conditional effects.

# Reference

Antonelli, J., Papadogeorgou, G. and Dominici, F. (2022) 'Causal Inference in High Dimensions: A Marriage Between Bayesian Modeling and Good Frequentist Properties,' *Biometrics*, 78(1), pp. 100-114. Available at: https://doi.org/10.1111/biom.13417.
"""
mutable struct BayesDRModel <: AbstractBDMLModel
    Y::Vector{Float64}
    T::Vector{Float64}
    X::Matrix{Float64}
    stats::StandardizationStats
    n::Int
    p::Int
    treatment_type::Symbol
    result::Union{Nothing, AbstractBDMLResult}
    is_fitted::Bool
    last_method::Union{Nothing, AbstractInferenceMethod}
end

function _validate_bayes_dr_data(Y, T, X, treatment_type)
    n = length(Y)
    n > 1 || throw(ArgumentError("BayesDRModel requires at least two observations"))
    length(T) == n || throw(DimensionMismatch("T must have the same length as Y"))
    size(X, 1) == n || throw(DimensionMismatch("X must have one row per observation"))
    size(X, 2) > 0 || throw(ArgumentError("BayesDRModel requires at least one covariate"))
    all(isfinite, Y) || throw(ArgumentError("Y must contain only finite values"))
    all(isfinite, T) || throw(ArgumentError("T must contain only finite values"))
    all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
    treatment_type in (:binary, :continuous) ||
        throw(ArgumentError("treatment_type must be :auto, :binary, or :continuous"))
    if treatment_type === :binary
        all(t -> t == 0 || t == 1, T) || throw(ArgumentError("binary T must be coded as 0/1"))
        length(unique(T)) == 2 || throw(ArgumentError("binary T must contain both treatment levels"))
    else
        length(unique(T)) > 2 ||
            throw(ArgumentError("continuous T must contain more than two distinct values"))
        std(T) > 0 || throw(ArgumentError("continuous T must have positive variance"))
    end
    return nothing
end

function BayesDRModel(Y, T, X; treatment_type::Symbol = :auto)
    resolved_treatment_type = if treatment_type === :auto
        all(t -> t == 0 || t == 1, T) ? :binary : :continuous
    else
        treatment_type
    end
    _validate_bayes_dr_data(Y, T, X, resolved_treatment_type)

    Y_f = Vector{Float64}(Y)
    T_f = Vector{Float64}(T)
    X_f = Matrix{Float64}(X)
    n, p = size(X_f)

    X_mean = vec(mean(X_f; dims = 1))
    X_sd = vec(std(X_f; dims = 1))
    all(isfinite, X_sd) && all(>(0), X_sd) ||
        throw(ArgumentError("X must not contain constant columns"))
    X_s = (X_f .- X_mean') ./ X_sd'

    Y_mean = mean(Y_f)
    Y_sd = std(Y_f)
    T_mean = mean(T_f)
    T_sd = std(T_f)
    stats = StandardizationStats(Y_mean, Y_sd, T_mean, T_sd, X_mean, X_sd)
    return BayesDRModel(
        Y_f, T_f, X_s, stats, n, p, resolved_treatment_type, nothing, false, nothing,
    )
end

function BayesDRModel(
        df::DataFrame, y::Symbol, treatment::Symbol;
        x_cols = nothing, treatment_type::Symbol = :auto,
    )
    columns = if x_cols === nothing
        filter(c -> c != y && c != treatment, Symbol.(names(df)))
    else
        Symbol.(x_cols)
    end
    return BayesDRModel(
        df[!, y], df[!, treatment], Matrix(df[:, columns]); treatment_type,
    )
end

model_type(::BayesDRModel) = :bayes_dr

function Base.show(io::IO, model::BayesDRModel)
    fitted = isfitted(model) ? "fitted" : "not fitted"
    println(io, "BayesDRModel ($fitted)")
    estimand = model.treatment_type === :binary ?
        "ATE (binary treatment)" : "Exposure-response curve E[Y(t)]"
    println(io, "  Estimand: $estimand")
    println(io, "  Observations: $(nobs(model))")
    return println(io, "  Covariates: $(ncovariates(model))")
end

Base.show(io::IO, ::MIME"text/plain", model::BayesDRModel) = show(io, model)

"""
    BayesDRMCMCMethod

Validated inference configuration produced by [`BayesDRMCMC`](@ref). Sampling
controls are arguments to [`fit!`](@ref), not fields of this type.
"""
struct BayesDRMCMCMethod <: AbstractInferenceMethod
    propensity_lower::Float64
    propensity_upper::Float64
    sigma_shape::Float64
    sigma_scale::Float64
    inclusion_a::Float64
    inclusion_b_scale::Float64
    slab_shape::Float64
    slab_scale::Float64
    curve_degree::Int
    density_ratio_lower::Float64
    density_ratio_upper::Float64
end

"""
    BayesDRMCMC(; propensity_bounds=(0.01, 0.99), sigma_shape=0.001,
               sigma_scale=0.001, inclusion_a=2.0,
               inclusion_b_scale=1.0, slab_shape=0.5, slab_scale=0.5,
               curve_degree=3, density_ratio_bounds=(1e-5, 1e5))

Construct the Gibbs inference method for [`BayesDRModel`](@ref). The
hyperparameters match the weak inverse-gamma and beta priors used by the
linear reference implementation of Antonelli et al. (2022).

# Keywords

- `propensity_bounds`: Lower and upper bounds applied to posterior propensity
  predictions before evaluating the doubly robust score.
- `sigma_shape`, `sigma_scale`: Shape and scale of the inverse-gamma prior for
  the outcome residual variance.
- `inclusion_a`: First shape parameter of each model's beta prior on its common
  covariate-inclusion probability.
- `inclusion_b_scale`: Multiplier of `p` used as the second beta-prior shape
  parameter.
- `slab_shape`, `slab_scale`: Shape and scale of the inverse-gamma prior for
  the slab variance.
- `curve_degree`: Polynomial degree used for a continuous-treatment
  exposure-response curve.
- `density_ratio_bounds`: Bounds for the stabilized density ratio in the
  continuous-treatment pseudo-outcome.

Sampling controls such as `n_samples`, `n_burn`, `thin`, `n_chains`, and
`n_boot` are passed to [`fit!`](@ref).

See Antonelli et al. (2022), Section 4 and Supporting Information, for the
spike-and-slab nuisance-model specification.
"""
function BayesDRMCMC(;
        propensity_bounds::Tuple{<:Real, <:Real} = (0.01, 0.99),
        sigma_shape::Real = 0.001,
        sigma_scale::Real = 0.001,
        inclusion_a::Real = 2.0,
        inclusion_b_scale::Real = 1.0,
        slab_shape::Real = 0.5,
        slab_scale::Real = 0.5,
        curve_degree::Int = 3,
        density_ratio_bounds::Tuple{<:Real, <:Real} = (1.0e-5, 1.0e5),
    )
    lower, upper = Float64.(propensity_bounds)
    0 <= lower < upper <= 1 || throw(ArgumentError("propensity bounds must satisfy 0 ≤ lower < upper ≤ 1"))
    sigma_shape > 0 || throw(ArgumentError("sigma_shape must be positive"))
    sigma_scale > 0 || throw(ArgumentError("sigma_scale must be positive"))
    inclusion_a > 0 || throw(ArgumentError("inclusion_a must be positive"))
    inclusion_b_scale > 0 || throw(ArgumentError("inclusion_b_scale must be positive"))
    slab_shape > 0 || throw(ArgumentError("slab_shape must be positive"))
    slab_scale > 0 || throw(ArgumentError("slab_scale must be positive"))
    curve_degree > 0 || throw(ArgumentError("curve_degree must be positive"))
    density_lower, density_upper = Float64.(density_ratio_bounds)
    0 < density_lower < density_upper ||
        throw(ArgumentError("density ratio bounds must satisfy 0 < lower < upper"))
    return BayesDRMCMCMethod(
        lower, upper, sigma_shape, sigma_scale, inclusion_a,
        inclusion_b_scale, slab_shape, slab_scale, curve_degree,
        density_lower, density_upper,
    )
end

uses_sampling(::BayesDRMCMCMethod) = true
supports_subsampling(::BayesDRMCMCMethod) = false
default_n_samples(::BayesDRMCMCMethod) = 1000
default_n_iterations(::BayesDRMCMCMethod) = 500

"""
    BayesDRNuisancePosterior

Stored draws from one or more nuisance-model chains. Rows correspond to
retained MCMC draws. For the treatment model, `coefficients` contains the
intercept followed by the covariate coefficients. For the outcome model, it
contains the intercept, one or more treatment-basis coefficients, and the
covariate coefficients.

`inclusion` contains only covariate inclusion indicators; the intercept and
outcome treatment-basis coefficients are always included. `chain_id`
identifies the source chain after chain results have been concatenated and is
used by `ess`, `rhat`, `mcse`, and `chain_info`.
"""
struct BayesDRNuisancePosterior
    coefficients::Matrix{Float64}
    inclusion::BitMatrix
    residual_variance::Vector{Float64}
    slab_variance::Vector{Float64}
    inclusion_probability::Vector{Float64}
    chain_id::Vector{Int}
end

"""
    BayesDRResult

Result of posterior-averaged doubly robust ATE estimation. `standard_error` is
the square root of the sum of the empirical-bootstrap variance and posterior
nuisance-parameter correction; it is not the standard deviation of
`posterior_effects`.

`posterior_effects` stores the AIPW estimate for every joint nuisance-posterior
draw. `bootstrap_estimates` stores estimates from the empirical bootstrap used
for the naive sampling variance. The reported interval combines these two
centered uncertainty components and is a posterior-corrected frequentist
confidence interval, not a Bayesian credible interval.

This follows the posterior-averaged doubly robust estimator in Equations 1-2
and the variance correction in Equation 4 of Antonelli et al. (2022).

Covariate inclusion probabilities can be recovered as
`vec(mean(result.treatment_posterior.inclusion; dims=1))` and similarly for
`result.outcome_posterior`.

# Reference

Antonelli, J., Papadogeorgou, G., & Dominici, F. (2022). Causal inference in
high dimensions: A marriage between Bayesian modeling and good frequentist
properties. *Biometrics, 78*(1), 100-114.
https://doi.org/10.1111/biom.13417
"""
struct BayesDRResult <: AbstractBDMLResult
    estimate::Float64
    standard_error::Float64
    confidence_interval::Tuple{Float64, Float64}
    level::Float64
    naive_variance::Float64
    posterior_variance::Float64
    posterior_effects::Vector{Float64}
    bootstrap_estimates::Vector{Float64}
    propensity_mean::Vector{Float64}
    propensity_clipped_fraction::Float64
    treatment_posterior::BayesDRNuisancePosterior
    outcome_posterior::BayesDRNuisancePosterior
    propensity_bounds::Tuple{Float64, Float64}
end

"""
    BayesDRCurveResult

Posterior-averaged doubly robust exposure-response curve for a continuous
treatment. Rows of `posterior_curves` and `bootstrap_curves` are curve draws;
columns correspond to `treatment_grid`. Confidence intervals are pointwise.

Use `ess`, `rhat`, `mcse`, and `chain_info` to inspect conservative summaries
across both nuisance-model chains.
"""
struct BayesDRCurveResult <: AbstractBDMLResult
    treatment_grid::Vector{Float64}
    estimate::Vector{Float64}
    standard_error::Vector{Float64}
    confidence_interval::Matrix{Float64}
    level::Float64
    naive_covariance::Matrix{Float64}
    posterior_covariance::Matrix{Float64}
    posterior_curves::Matrix{Float64}
    bootstrap_curves::Matrix{Float64}
    density_ratio_clipped_fraction::Float64
    treatment_posterior::BayesDRNuisancePosterior
    outcome_posterior::BayesDRNuisancePosterior
    curve_degree::Int
    density_ratio_bounds::Tuple{Float64, Float64}
end

function extract_alpha(::BayesDRResult)
    return error(
        "BayesDRResult estimates an ATE (Δ), not posterior samples of α. " *
            "Use coef(result), stderror(result), or confint(result)."
    )
end

function extract_alpha(::BayesDRCurveResult)
    return error(
        "BayesDRCurveResult estimates E[Y(t)] over a treatment grid, not " *
            "posterior samples of α. Use coef(result), stderror(result), or confint(result)."
    )
end

function credible_interval(::BayesDRResult; level = 0.95)
    return error(
        "BayesDRResult has a posterior-corrected confidence interval, not a " *
            "posterior credible interval. Use confint(result; level=$level)."
    )
end

function credible_interval(::BayesDRCurveResult; level = 0.95)
    return error(
        "BayesDRCurveResult has posterior-corrected pointwise confidence " *
            "intervals, not posterior credible intervals. Use confint(result; level=$level)."
    )
end

export BayesDRModel, BayesDRMCMCMethod, BayesDRMCMC, BayesDRResult, BayesDRCurveResult
