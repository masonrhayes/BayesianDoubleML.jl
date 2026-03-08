# Data Generating Processes for Bayesian Double Machine Learning
# Implements the simulation designs from DiTraglia & Liu (2025)

module DGP

using Random
using LinearAlgebra
using Statistics

export generate_dgp_table1

"""
    generate_dgp_table1(n::Int, p::Int, sigma_epsilon::Real; seed=nothing)

Generate synthetic data matching the simulation design from 
DiTraglia & Liu (2025), Section 6, Table 1.

This DGP corresponds to the "fixed" design described in Equations (20)-(21)
of the paper, which is used to generate the results reported in Table 1.

# Data Generating Process (Equations 20-21)

**Covariates:**
    X_i ~ iid Normal_p(0, I_p)

**Errors:**
    (ε_i, V_i)' | X ~ iid Normal([0; 0], [σ²_ε, 0; 0, 1])
    
where:
    - ε_i: Structural error (uncorrelated with V_i)
    - V_i: Treatment error (variance = 1)
    - σ_ε: Standard deviation of ε (varies across simulations)

**Coefficient distribution:**
    β ~ Normal_p(μ_β, σ²_β · I_p)

**Fixed parameters (per paper):**
    - α = 2                          (true causal effect)
    - γ = ι_p / √p                   (treatment coefficients)
    - μ_β = -γ/2                     (mean of outcome coefficients)
    - σ²_β = 1/p                     (variance of outcome coefficients)

**Construction:**
    D_i = X_i'γ + V_i                (Equation 4: Treatment reduced form)
    Y_i = α·D_i + X_i'β + ε_i       (Equation 5: Outcome structural)

# Arguments
- `n::Int`: Number of observations (paper uses: 200)
- `p::Int`: Number of covariates (paper uses: 100)
- `sigma_epsilon::Real`: Std dev of structural error ε ∈ {1, 2, 4}
- `seed::Union{Int,Nothing}`: Random seed for reproducibility (optional)

# Returns
- `Y::Vector{Float64}`: Outcome variable (length n)
- `D::Vector{Float64}`: Treatment variable (length n)
- `X::Matrix{Float64}`: Covariates (n×p matrix)
- `alpha_true::Float64`: True causal effect (always 2.0)
- `params::NamedTuple`: Ground truth parameters for validation/testing:
  - `gamma::Vector{Float64}`: Treatment coefficients (γ = ι_p/√p)
  - `beta::Vector{Float64}`: Outcome coefficients (drawn from N(μ_β, σ²_β·I))
  - `mu_beta::Vector{Float64}`: Mean of β distribution (-γ/2)
  - `sigma2_beta::Float64`: Variance of β distribution (1/p)
  - `V::Vector{Float64}`: Treatment errors
  - `epsilon::Vector{Float64}`: Structural errors

# Paper Reference
Section 6, "Simulation Study", Equations (20)-(21):
> DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
> Causal Inference", arXiv:2508.12688v1.

# Table 1 Settings
The paper reports results for three values of σ_ε:

| σ_ε | BDML-Hier Coverage | BDML-Hier RMSE | BDML-Basic Coverage | BDML-Basic RMSE |
|-----|-------------------|----------------|---------------------|-----------------|
| 1   | 0.94              | 0.09           | 0.93                | 0.11            |
| 2   | 0.94              | 0.18           | 0.91                | 0.22            |
| 4   | 0.94              | 0.35           | 0.92                | 0.46            |

All settings use n=200, p=100, and α=2.

# Examples

## Basic usage
```julia
# Replicate one row of Table 1 (n=200, p=100, σ_ε=2)
Y, D, X, alpha_true, params = generate_dgp_table1(200, 100, 2.0; seed=42)

# Verify dimensions
@assert length(Y) == 200
@assert length(D) == 200
@assert size(X) == (200, 100)
@assert alpha_true == 2.0
```

## Generate all three σ_ε settings
```julia
for σ in [1.0, 2.0, 4.0]
    Y, D, X, alpha_true, params = generate_dgp_table1(200, 100, σ; seed=123)
    println("σ_ε = " * string(σ) * ": generated " * string(length(Y)) * " observations")
end
```

## Use with BDML
```julia
using BayesianDoubleML

# Generate data
Y, D, X, alpha_true, params = generate_dgp_table1(200, 100, 2.0; seed=42)

# Create problem and fit
problem = BDMLProblem(Y, D, X; model_type=:hier)
result = fit(problem, MCMCNUTS(); n_samples=1000, n_chains=1)

# Check if we recover α
estimated_alpha = mean(result.alpha_samples)
println("True α: \$alpha_true, Estimated α: \$estimated_alpha")
```

# Implementation Notes

**Randomness:**
- The function generates new random draws for X, β, ε, and V in each call
- Setting `seed` ensures reproducibility
- Following the paper, each replication should use a fresh seed or different seed

**Coefficient generation:**
- γ is fixed at ι_p/√p (as specified in paper)
- β is randomly drawn from N(μ_β, σ²_β·I) for each replication
- This mimics the paper's design where β varies across replications

**Error structure:**
- ε and V are independent (covariance matrix is diagonal)
- Var(V) = 1 (normalized)
- Var(ε) = σ²_ε (varies across simulation settings)

**Confounding:**
- X affects both D (through γ) and Y (through β)
- This creates confounding that BDML is designed to handle
- The specific structure (γ = ι_p/√p, μ_β = -γ/2) creates realistic correlation

See also: [`generate_dgp_table1`](@ref)
"""
function generate_dgp_table1(
        n::Int, p::Int, sigma_epsilon::Real; alpha_true = 2.0, rng = Random.default_rng()
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
    Y = alpha_true .* D + X * beta + epsilon

    # Package ground truth parameters for validation/testing
    params = (
        gamma = gamma,
        beta = beta,
        mu_beta = mu_beta,
        sigma2_beta = sigma2_beta,
        V = V,
        epsilon = epsilon,
    )

    return Y, D, X, alpha_true, params
end

"""
    generate_dgp_table1(; n=200, p=100, sigma_epsilon=2.0, seed=nothing)

Convenience method with default parameters matching Table 1 of the paper.

# Default Parameters
- n = 200 (number of observations)
- p = 100 (number of covariates)  
- sigma_epsilon = 2.0 (middle value from Table 1)
- seed = nothing (random)

# Example
```julia
# Generate standard Table 1 data
Y, D, X, alpha_true, params = generate_dgp_table1()

# All defaults: n=200, p=100, σ_ε=2.0
```
"""
function generate_dgp_table1(; n::Int = 200, p::Int = 100, sigma_epsilon::Real = 2.0, seed::Union{Int, Nothing} = nothing)
    return generate_dgp_table1(n, p, sigma_epsilon; seed = seed)
end

end # module DGP
