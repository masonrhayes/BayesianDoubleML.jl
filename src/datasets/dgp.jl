# Data Generating Processes for Bayesian Double Machine Learning
# Implements the simulation designs from DiTraglia & Liu (2025)

module DGP

using Random
using LinearAlgebra
using Statistics
using DataFrames
using Distributions

export make_plr_DTL2025, make_irm_APD2022, make_er_APD2022, make_plr_LML2025

"""
    make_plr_DTL2025(n::Int, p::Int, sigma_epsilon::Real;
                     alpha=2.0, rng=Random.default_rng())
    make_plr_DTL2025(rng::AbstractRNG; n, p, sigma_epsilon, alpha=2.0)

Generate synthetic data matching the simulation design from 
DiTraglia & Liu (2025), Section 6, Table 1.

This DGP corresponds to the "fixed" design described in Equations (20)-(21)
of the paper, which is used to generate the results reported in Table 1.

# Data Generating Process (Equations 20-21)

**Covariates:**
    ``X_i \\sim \\text{iid Normal}_p(0, I_p)``

**Errors:**
    ``(\\epsilon_i, V_i)' | X \\sim \\text{iid Normal}([0; 0], [\\sigma^2_\\epsilon, 0; 0, 1])``
    
where:
-  ``\\epsilon_i``: Structural error (uncorrelated with ``V_i``)
-  ``V_i``: Treatment error (variance = 1)
-  ``\\sigma_\\epsilon``: Standard deviation of ``\\epsilon`` (varies across simulations)

**Coefficient distribution:**
    ``\\beta \\sim \\text{Normal}_p(\\mu_\\beta, \\sigma^2_\\beta \\cdot I_p)``

**Fixed parameters (per paper):**
-   ``\\alpha = 2``                          (true causal effect)
-   ``\\gamma = \\iota_p / \\sqrt{p}``                   (treatment coefficients)
-   ``\\mu_\\beta = -\\gamma/2``                     (mean of outcome coefficients)
-   ``\\sigma^2_\\beta = 1/p``                     (variance of outcome coefficients)

**Construction:**
-  ``D_i = X_i'\\gamma + V_i``                (Equation 4: Treatment reduced form)
-  ``Y_i = \\alpha \\cdot D_i + X_i'\\beta + \\epsilon_i``       (Equation 5: Outcome structural)

# Arguments
- `n::Int`: Number of observations (paper uses: 200)
- `p::Int`: Number of covariates (paper uses: 100)
- `sigma_epsilon::Real`: Std dev of structural error ``\\epsilon \\in \\{1, 2, 4\\}``
- `alpha::Real`: True causal effect (default: 2.0)
- `rng::AbstractRNG`: Random number generator for reproducibility (default: `Random.default_rng()`)

# Returns
- `df::DataFrame`: DataFrame containing:
  - `y::Vector{Float64}`: Outcome variable (length n)
  - `d::Vector{Float64}`: Treatment variable (length n)
  - `X1, X2, ..., Xp::Vector{Float64}`: Covariates as columns

Extract the components for use with `BDMLModel`:
```julia
df = make_plr_DTL2025(n, p, sigma_epsilon; alpha=2.0, rng)
```

# Paper Reference
Section 6, "Simulation Study", Equations (20)-(21):
> DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
> Causal Inference", arXiv:2508.12688v1.

# Table 1 Settings
The paper reports results for three values of ``\\sigma_\\epsilon``:

| ``\\sigma_\\epsilon`` | BDML-Hier Coverage | BDML-Hier RMSE | BDML-Basic Coverage | BDML-Basic RMSE |
|-----|-------------------|----------------|---------------------|-----------------|
| 1   | 0.94              | 0.09           | 0.93                | 0.11            |
| 2   | 0.94              | 0.18           | 0.91                | 0.22            |
| 4   | 0.94              | 0.35           | 0.92                | 0.46            |

All settings use n=200, p=100, and ``\\alpha=2``.

# Examples

## Basic usage
```julia
# Replicate one row of Table 1 (n=200, p=100, σ_ε=2)
using BayesianDoubleML
using Random

df = make_plr_DTL2025(200, 100, 2.0; rng=MersenneTwister(42))

```

# Implementation Notes

**Randomness:**
- The function generates new random draws for ``X``, ``\beta``, ``\epsilon``, and ``V`` in each call
- Pass an `rng` argument (e.g., `MersenneTwister(seed)`) for reproducibility
- Following the paper, each replication should use a fresh seed or different seed

**Coefficient generation:**
- ``\\gamma`` is fixed at ``\\iota_p/\\sqrt{p}`` (as specified in paper)
- ``\\beta`` is randomly drawn from ``N(\\mu_\\beta, \\sigma^2_\\beta\\cdot I)`` for each replication
- This mimics the paper's design where ``\\beta`` varies across replications

**Error structure:**
- ``\\epsilon`` and ``V`` are independent (covariance matrix is diagonal)
- ``\\text{Var}(V) = 1`` (normalized)
- ``\\text{Var}(\\epsilon) = \\sigma^2_\\epsilon`` (varies across simulation settings)

**Confounding:**
- ``X`` affects both ``D`` (through ``\\gamma``) and ``Y`` (through ``\\beta``)
- This creates confounding that BDML is designed to handle
- The specific structure (``\\gamma = \\iota_p/\\sqrt{p}``, ``\\mu_\\beta = -\\gamma/2``) creates realistic correlation

See also: [`make_plr_DTL2025`](@ref)
"""
function make_plr_DTL2025(
        n::Int, p::Int, sigma_epsilon::Real; alpha = 2.0, rng = Random.default_rng()
    )

    # Validate inputs
    @assert n > 0 "n must be positive"
    @assert p > 0 "p must be positive"
    @assert sigma_epsilon > 0 "sigma_epsilon must be positive"

    # Fixed parameters from paper (Equation 21)
    gamma = ones(p) ./ sqrt(p)          # γ = ι_p / √p  (treatment coefficients)
    mu_beta = -gamma ./ 2              # μ_β = -γ/2    (mean of outcome coefficients)
    sigma2_beta = 1.0 / p              # σ²_β = 1/p    (variance of outcome coefficients)

    # Generate covariates (Equation 20): X_i ~ N(0, I_p)
    X = randn(rng, n, p)

    # Generate outcome coefficients (Equation 20): β ~ N(μ_β, σ²_β · I_p)
    # These are drawn fresh for each replication as per paper
    beta = randn(rng, p) .* sqrt(sigma2_beta) .+ mu_beta

    # Generate errors (Equation 20)
    V = randn(rng, n)                                  # Treatment error: V ~ N(0, 1)
    epsilon = randn(rng, n) .* sigma_epsilon           # Structural error: ε ~ N(0, σ²_ε)

    # Construct treatment D (Equation 4): D = X'γ + V
    D = X * gamma + V

    # Construct outcome Y (Equation 5): Y = α·D + X'β + ε
    Y = alpha .* D + X * beta + epsilon

    # Package ground truth parameters for validation/testing
    params = (
        gamma = gamma,
        beta = beta,
        mu_beta = mu_beta,
        sigma2_beta = sigma2_beta,
        V = V,
        epsilon = epsilon,
    )

    df = DataFrame(X, [Symbol("X$i") for i in 1:p])
    df.y = Y
    df.d = D

    return df
end

function make_plr_DTL2025(
        rng::AbstractRNG;
        n::Int,
        p::Int,
        sigma_epsilon::Real,
        alpha::Real = 2.0,
    )
    return make_plr_DTL2025(n, p, sigma_epsilon; alpha, rng)
end

"""
    make_irm_APD2022([n=100, p=500]; scenario=:linear, alpha=1.0,
                     rng=Random.default_rng())
    make_irm_APD2022(rng::AbstractRNG; n=100, p=500,
                     scenario=:linear, alpha=1.0)

Generate the binary-treatment simulation design from Antonelli,
Papadogeorgou, and Dominici (2022), Section 5.1.

The covariates follow an equicorrelated Gaussian distribution with unit
variances and pairwise correlation 0.3. The treatment uses a probit propensity
model. `scenario=:linear` generates

```math
P(D_i=1\\mid X_i)=\\Phi(0.15X_{i1}+0.2X_{i2}-0.4X_{i5})
```

and

```math
E(Y_i\\mid D_i,X_i)=\\alpha D_i+0.75X_{i1}+X_{i2}+0.6X_{i3}
                    -0.8X_{i4}-0.7X_{i5}.
```

For `scenario=:nonlinear`, the treatment and outcome predictors are

```math
P(D_i=1\\mid X_i)=\\Phi(0.15X_{i1}-0.4X_{i2}-0.5X_{i5}),
```

```math
E(Y_i\\mid D_i,X_i)=\\alpha D_i+0.8X_{i1}+0.4X_{i2}^3
  +0.25\\exp(|X_{i2}|)+0.8X_{i5}^2-1.5\\sin(X_{i5}).
```

In both scenarios the outcome error is standard normal and the true ATE is
`alpha`.

# Returns
A `DataFrame` with columns `X1` through `Xp`, followed by `y` and binary `d`.

# Reference
Antonelli, J., Papadogeorgou, G., & Dominici, F. (2022). Causal inference in
high dimensions: A marriage between Bayesian modeling and good frequentist
properties. *Biometrics, 78*(1), 100-114. https://doi.org/10.1111/biom.13417
"""
function make_irm_APD2022(
        n::Int = 100, p::Int = 500;
        scenario::Symbol = :linear,
        alpha::Real = 1.0,
        rng::AbstractRNG = Random.default_rng(),
    )
    n > 0 || throw(ArgumentError("n must be positive"))
    p >= 5 || throw(ArgumentError("p must be at least 5 for the APD2022 design"))
    scenario in (:linear, :nonlinear) ||
        throw(ArgumentError("scenario must be :linear or :nonlinear"))
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))

    common_factor = randn(rng, n)
    X = sqrt(0.3) .* reshape(common_factor, n, 1) .+ sqrt(0.7) .* randn(rng, n, p)

    treatment_predictor = if scenario === :linear
        @. 0.15 * X[:, 1] + 0.2 * X[:, 2] - 0.4 * X[:, 5]
    else
        @. 0.15 * X[:, 1] - 0.4 * X[:, 2] - 0.5 * X[:, 5]
    end
    propensity = cdf.(Normal(), treatment_predictor)
    D = Float64.(rand(rng, n) .< propensity)

    outcome_predictor = if scenario === :linear
        @. alpha * D + 0.75 * X[:, 1] + X[:, 2] + 0.6 * X[:, 3] -
            0.8 * X[:, 4] - 0.7 * X[:, 5]
    else
        @. alpha * D + 0.8 * X[:, 1] + 0.4 * X[:, 2]^3 +
            0.25 * exp(abs(X[:, 2])) + 0.8 * X[:, 5]^2 - 1.5 * sin(X[:, 5])
    end
    Y = outcome_predictor + randn(rng, n)

    df = DataFrame(X, [Symbol("X$i") for i in 1:p])
    df.y = Y
    df.d = D
    return df
end

function make_irm_APD2022(
        rng::AbstractRNG;
        n::Int = 100,
        p::Int = 500,
        scenario::Symbol = :linear,
        alpha::Real = 1.0,
    )
    return make_irm_APD2022(n, p; scenario, alpha, rng)
end

"""
    make_er_APD2022([rng::AbstractRNG]; n=200, p=200,
                    cubic_coefficient=0.05, quadratic_coefficient=-0.1)

Generate the continuous-treatment exposure-response simulation design from
Antonelli, Papadogeorgou, and Dominici (2022), Section 5.2.

The covariates follow an equicorrelated Gaussian distribution with unit
variances and pairwise correlation 0.3. Treatment and outcome are generated as

```math
T_i \\mid X_i \\sim N(\\mu_i^t, 1),
```

```math
\\mu_i^t = 0.6X_{i1} + 0.6X_{i2} + \\exp(0.65|X_{i1}|) - 0.8X_{i3}^2,
```

```math
Y_i \\mid T_i,X_i \\sim N(\\mu_i^y, 1),
```

```math
\\mu_i^y = 5 + 0.05T_i^3 - 0.1T_i^2 + 0.6X_{i1}
          + 0.4\\exp(X_{i1}) + \\log(0.65|X_{i2}|)
          + 0.5(1 + X_{i3})^2.
```

The true exposure-response curve is cubic in treatment, up to an additive
constant obtained by marginalizing over the covariates. The signed
`cubic_coefficient` and `quadratic_coefficient` keywords customize its shape;
their defaults reproduce the paper.

# Returns
A `DataFrame` with columns `X1` through `Xp`, followed by `y` and continuous
`d`.

# Reference
Antonelli, J., Papadogeorgou, G., & Dominici, F. (2022). Causal inference in
high dimensions: A marriage between Bayesian modeling and good frequentist
properties. *Biometrics, 78*(1), 100-114. https://doi.org/10.1111/biom.13417
"""
function make_er_APD2022(
        rng::AbstractRNG = Random.default_rng();
        n::Int = 200,
        p::Int = 200,
        cubic_coefficient::Real = 0.05,
        quadratic_coefficient::Real = -0.1,
    )
    n > 0 || throw(ArgumentError("n must be positive"))
    p >= 3 || throw(ArgumentError("p must be at least 3 for the APD2022 design"))
    isfinite(cubic_coefficient) || throw(ArgumentError("cubic_coefficient must be finite"))
    isfinite(quadratic_coefficient) ||
        throw(ArgumentError("quadratic_coefficient must be finite"))

    common_factor = randn(rng, n)
    X = sqrt(0.3) .* reshape(common_factor, n, 1) .+ sqrt(0.7) .* randn(rng, n, p)

    treatment_mean = @. 0.6 * X[:, 1] + 0.6 * X[:, 2] +
        exp(0.65 * abs(X[:, 1])) - 0.8 * X[:, 3]^2
    D = treatment_mean + randn(rng, n)

    outcome_mean = @. 5 + cubic_coefficient * D^3 + quadratic_coefficient * D^2 +
        0.6 * X[:, 1] +
        0.4 * exp(X[:, 1]) + log(0.65 * abs(X[:, 2])) +
        0.5 * (1 + X[:, 3])^2
    Y = outcome_mean + randn(rng, n)
    all(isfinite, Y) || error("APD2022 outcome generation produced a non-finite value")

    df = DataFrame(X, [Symbol("X$i") for i in 1:p])
    df.y = Y
    df.d = D
    return df
end

function make_er_APD2022(
        n::Int, p::Int = 200;
        cubic_coefficient::Real = 0.05,
        quadratic_coefficient::Real = -0.1,
        rng::AbstractRNG = Random.default_rng(),
    )
    return make_er_APD2022(
        rng; n, p, cubic_coefficient, quadratic_coefficient,
    )
end

"""
    make_plr_LML2025([rng::AbstractRNG]; n=40, p=40, treatment=:continuous,
                     alpha=1.0)

Generate a simulation design from Luo et al. (2025). The default
`treatment = :continuous` reproduces Section 3.2:

```math
X_i \\sim N_p(0, \\Sigma), \\qquad
\\Sigma_{jk} = \\begin{cases}1 & j=k, \\\\ 0.05 & j \\ne k,\\end{cases}
```

```math
D_i \\mid X_i \\sim N(0.45X_{i1}+0.9X_{i2}-0.4X_{i5}, 1),
```

```math
Y_i \\mid D_i,X_i \\sim
N(\\alpha D_i+0.5X_{i1}+X_{i3}-0.1X_{i4}-0.2X_{i7}, 1).
```

The treatment coefficient is controlled by `alpha`, which defaults to one.
The paper uses `n = 40` and `p = 40`.

With `treatment = :binary`, the function reproduces Section 3.1:

```math
X_i \\sim N_p(0, \\Sigma), \\qquad
\\Sigma_{jk} = \\begin{cases}1 & j=k, \\\\ 0.3 & j \\ne k,\\end{cases}
```

```math
D_i \\mid X_i \\sim
\\operatorname{Bernoulli}(\\operatorname{expit}(0.3X_{i1}+0.2X_{i2}-0.4X_{i5})).
```

The outcome equation is the same as in the continuous design and the true ATE
is `alpha`. The paper uses `p = 500` and `n` equal to either 50 or 200 for this
design.

# Returns
A `DataFrame` with columns `X1` through `Xp`, followed by `y` and `d`.

# Reference
Luo, M., Moodie, E. E. M., Bhatnagar, S., & Lee, D. (2025). A scalable
Bayesian double machine learning framework, with application to racial
disproportionality. Sections 3.1-3.2.
"""
function make_plr_LML2025(
        rng::AbstractRNG = Random.default_rng();
        n::Int = 40,
        p::Int = 40,
        treatment::Symbol = :continuous,
        alpha::Real = 1.0,
    )
    n > 0 || throw(ArgumentError("n must be positive"))
    p >= 7 || throw(ArgumentError("p must be at least 7 for the LML2025 design"))
    treatment in (:continuous, :binary) ||
        throw(ArgumentError("treatment must be :continuous or :binary"))
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    return _make_plr_LML2025(rng, n, p, Val(treatment), Float64(alpha))
end

function _make_plr_LML2025(
        rng::AbstractRNG, n::Int, p::Int, ::Val{:continuous}, alpha::Float64,
    )
    Σ = fill(0.05, p, p)
    Σ[diagind(Σ)] .= 1.0

    X = permutedims(rand(rng, MvNormal(zeros(p), Σ), n))

    μ_D = 0.45 .* X[:, 1] .+ 0.9 .* X[:, 2] .- 0.4 .* X[:, 5]
    D = μ_D .+ randn(rng, n)

    μ_Y = alpha .* D .+ 0.5 .* X[:, 1] .+ X[:, 3] .- 0.1 .* X[:, 4] .- 0.2 .* X[:, 7]
    Y = μ_Y .+ randn(rng, n)

    df = DataFrame(X, [Symbol("X$i") for i in 1:p])
    df.y = Y
    df.d = D
    return df
end

function _make_plr_LML2025(
        rng::AbstractRNG, n::Int, p::Int, ::Val{:binary}, alpha::Float64,
    )
    Σ = fill(0.3, p, p)
    Σ[diagind(Σ)] .= 1.0

    X = permutedims(rand(rng, MvNormal(zeros(p), Σ), n))

    linear_predictor = 0.3 .* X[:, 1] .+ 0.2 .* X[:, 2] .- 0.4 .* X[:, 5]
    propensity = @. inv(1 + exp(-linear_predictor))
    D = Float64.(rand(rng, n) .< propensity)

    μ_Y = alpha .* D .+ 0.5 .* X[:, 1] .+ X[:, 3] .- 0.1 .* X[:, 4] .- 0.2 .* X[:, 7]
    Y = μ_Y .+ randn(rng, n)

    df = DataFrame(X, [Symbol("X$i") for i in 1:p])
    df.y = Y
    df.d = D
    return df
end

end # module DGP
