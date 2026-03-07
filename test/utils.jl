# Test utilities for BayesianDoubleML.jl
# Shared functions and data generation for all tests

using Random
using Statistics

"""
    make_test_data(; n=100, p=10, alpha_true=0.5, seed=123)

Generate synthetic test data for Double Machine Learning.

# Arguments
- `n::Int=100`: Number of observations
- `p::Int=10`: Number of covariates (control variables)
- `alpha_true::Float64=0.5`: True causal effect (α)
- `seed::Int=123`: Random seed for reproducibility

# Data Generating Process
- X ~ N(0, I) where X is n×p
- D = X * γ + ε₁ where γ ~ N(0, 0.5²·I), ε₁ ~ N(0, 0.5²)
- Y = α·D + X * β + ε₂ where β ~ N(0, 0.3²·I), ε₂ ~ N(0, 1)

This creates a scenario with confounding (X affects both D and Y).

# Returns
- `Y::Vector{Float64}`: Outcome variable (length n)
- `D::Vector{Float64}`: Treatment variable (length n)
- `X::Matrix{Float64}`: Control variables (n×p)
- `alpha_true::Float64`: True causal effect

# Examples
```julia
Y, D, X, alpha = make_test_data(n=500, p=20, alpha_true=2.0, seed=42)
```
"""
function make_test_data(; n::Int = 100, p::Int = 10, alpha_true::Float64 = 0.5, seed::Int = 123)
    Random.seed!(seed)

    # Covariates
    X = randn(n, p)

    # Treatment: confounded with X
    gamma = randn(p) .* 0.5
    D = X * gamma + randn(n) .* 0.5

    # Outcome: affected by treatment and covariates
    beta = randn(p) .* 0.3
    Y = alpha_true .* D + X * beta + randn(n)

    return Y, D, X, alpha_true
end

"""
    make_binary_treatment_data(; n=100, p=10, alpha_true=0.5, seed=123)

Generate synthetic test data with binary treatment (for IRM models).

Similar to `make_test_data` but with binary D ∈ {0, 1}.

# Arguments
Same as `make_test_data`.

# Data Generating Process
- X ~ N(0, I)
- propensity = logistic(X * γ) where γ ~ N(0, 0.5²·I)
- D ~ Bernoulli(propensity)
- Y = α·D + X * β + ε

# Returns
Same as `make_test_data`, but D is binary (0 or 1).
"""
function make_binary_treatment_data(; n::Int = 100, p::Int = 10, alpha_true::Float64 = 0.5, seed::Int = 123)
    Random.seed!(seed)

    X = randn(n, p)
    gamma = randn(p) .* 0.5

    # Propensity score (probability of treatment)
    propensity = 1.0 ./ (1.0 .+ exp.(-X * gamma))

    # Binary treatment
    D = Float64.(rand(n) .< propensity)

    # Outcome
    beta = randn(p) .* 0.3
    Y = alpha_true .* D + X * beta + randn(n)

    return Y, D, X, alpha_true
end

"""
    logistic(x)

Numerically stable logistic (sigmoid) function.

# Arguments
- `x::Real`: Input value

# Returns
- `Float64`: 1 / (1 + exp(-x))
"""
logistic(x::Real) = x < 0 ? exp(x) / (1 + exp(x)) : 1 / (1 + exp(-x))
