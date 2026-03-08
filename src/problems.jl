# BDML Problem Types for Multiple Dispatch
# Defines the "what" - the problem specification with pre-computed data

export AbstractBDMLProblem, BDMLBasicProblem, BDMLHierarchicalProblem, BDMLProblem
export nobs, ncovariates, model_type, standardization_stats

"""
    AbstractBDMLProblem

Abstract type for all BDML problem specifications.

Problem types encapsulate:
- Data (Y, D, X) - already standardized
- Standardization statistics (for transforming back)
- Pre-allocated temporaries (for performance)
- Metadata (n, p, model_type)

Each concrete subtype represents a different model specification
(basic vs hierarchical) and can be fitted with different methods
(MCMC, VI unified, VI simple) via multiple dispatch.
"""
abstract type AbstractBDMLProblem end

"""
    BDMLBasicProblem <: AbstractBDMLProblem

Basic BDML model problem specification implementing BDML-Basic from 
DiTraglia & Liu (2025), Algorithm 1.

# Fields
- `Y::Vector{Float64}`: Standardized outcome variable
- `D::Vector{Float64}`: Standardized treatment variable  
- `X::Matrix{Float64}`: Standardized control variables (covariates)
- `stats::StandardizationStats`: Statistics for transforming back to original scale
- `n::Int`: Number of observations
- `p::Int`: Number of control variables
- `μ_Y_cache::Vector{Float64}`: Pre-allocated temporary for outcome mean (X'δ)
- `μ_D_cache::Vector{Float64}`: Pre-allocated temporary for treatment mean (X'γ)

# Model Specification (Section 4, Equations 12-13)

**Reduced form:**
    Y = X'δ + U          (Eq. 12)
    D = X'γ + V          (Eq. 5)

**Priors:**
- δ ~ N(0, 25·I_p)      [Outcome reduced form coefficients]
- γ ~ N(0, 25·I_p)      [Treatment reduced form coefficients]  
- σ_U ~ Cauchy⁺(0, 2.5) [Outcome error scale]
- σ_V ~ Cauchy⁺(0, 2.5) [Treatment error scale]
- R ~ LKJ(4)            [Correlation matrix]

For VI, uses Beta(2,2) correlation parameterization instead of LKJCholesky.

**Causal effect recovery:**
    α = ρ·σ_U / σ_V       (Eq. 15)

This is the BDML-Basic variation with fixed prior variances (Section 6, Table 1).

See also: [`BDMLHierarchicalProblem`](@ref), [`BDMLProblem`](@ref)

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.
"""
struct BDMLBasicProblem <: AbstractBDMLProblem
    Y::Vector{Float64}
    D::Vector{Float64}
    X::Matrix{Float64}
    stats::StandardizationStats
    n::Int
    p::Int
    # Pre-allocated temporaries for performance
    μ_Y_cache::Vector{Float64}
    μ_D_cache::Vector{Float64}
end

"""
    BDMLHierarchicalProblem <: AbstractBDMLProblem

Hierarchical BDML model problem specification implementing BDML-Hier from 
DiTraglia & Liu (2025), Algorithm 1.

# Fields
Same as `BDMLBasicProblem`:
- `Y::Vector{Float64}`: Standardized outcome variable
- `D::Vector{Float64}`: Standardized treatment variable  
- `X::Matrix{Float64}`: Standardized control variables (covariates)
- `stats::StandardizationStats`: Standardization statistics
- `n::Int`: Number of observations
- `p::Int`: Number of control variables
- `μ_Y_cache::Vector{Float64}`: Pre-allocated temporary for outcome mean (X'δ)
- `μ_D_cache::Vector{Float64}`: Pre-allocated temporary for treatment mean (X'γ)

# Model Specification (Section 4, Equations 12-13)

**Reduced form:**
    Y = X'δ + U          (Eq. 12)
    D = X'γ + V          (Eq. 5)

**Hierarchical priors:**
- σ²_δ ~ InvGamma(2, 2)       [Variance hyperparameter for δ]
- σ²_γ ~ InvGamma(2, 2)       [Variance hyperparameter for γ]
- δ ~ N(0, σ²_δ·I_p)         [Outcome coefficients with adaptive shrinkage]
- γ ~ N(0, σ²_γ·I_p)         [Treatment coefficients with adaptive shrinkage]
- σ_U ~ Cauchy⁺(0, 2.5)       [Outcome error scale]
- σ_V ~ Cauchy⁺(0, 2.5)       [Treatment error scale]
- R ~ LKJ(4)                 [Correlation matrix]

For VI, uses Beta(2,2) correlation parameterization.

**Causal effect recovery:**
    α = ρ·σ_U / σ_V            (Eq. 15)

**Hierarchical structure:**
This is equivalent to placing independent Student-t(4) distributions on each 
coefficient marginally. The InvGamma(2, 2) hyperprior provides adaptive 
shrinkage that learns the appropriate regularization from data (Section 6, Table 1).

This can improve performance when p is large relative to n, and the paper's
simulations show BDML-Hier achieves better coverage (0.94) than BDML-Basic (0.91-0.93).

See also: [`BDMLBasicProblem`](@ref), [`BDMLProblem`](@ref)

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.
"""
struct BDMLHierarchicalProblem <: AbstractBDMLProblem
    Y::Vector{Float64}
    D::Vector{Float64}
    X::Matrix{Float64}
    stats::StandardizationStats
    n::Int
    p::Int
    # Pre-allocated temporaries for performance
    μ_Y_cache::Vector{Float64}
    μ_D_cache::Vector{Float64}
end

"""
    BDMLProblem(Y, D, X; model_type=:basic)

Factory function to create appropriate BDML problem type for Algorithm 1
from DiTraglia & Liu (2025).

Standardizes data once during creation and pre-allocates temporaries.
This ensures data is only standardized once, even if fitted multiple times.

# Arguments
- `Y::Vector{Float64}`: Outcome variable (will be standardized)
- `D::Vector{Float64}`: Treatment variable (will be standardized)
- `X::Matrix{Float64}`: Control variables (will be standardized)

# Keyword Arguments
- `model_type::Symbol=:basic`: 
  - `:basic` for BDML-Basic (fixed prior variances)
  - `:hier` for BDML-Hier (adaptive hierarchical priors)

# Returns
`AbstractBDMLProblem`: Either `BDMLBasicProblem` or `BDMLHierarchicalProblem`

# Algorithm 1 Variations

**BDML-Basic** (`model_type=:basic`):
- Places independent N(0, 25·I) priors on δ and γ
- Fixed shrinkage across all covariates
- Recommended for interpretability and simplicity

**BDML-Hier** (`model_type=:hier`):
- Places hierarchical InvGamma(2, 2) priors on σ²_δ and σ²_γ
- Adaptive shrinkage equivalent to Student-t(4) on coefficients
- Better coverage in simulations (Table 1: 0.94 vs 0.91-0.93)
- Recommended as default choice

Both variations recover α via Equation 15: α = ρ·σ_U / σ_V

# Examples
```julia
# Create basic model problem
prob_basic = BDMLProblem(Y, D, X; model_type=:basic)

# Create hierarchical model problem  
prob_hier = BDMLProblem(Y, D, X; model_type=:hier)

# Fit with different methods
result_mcmc = fit(prob_basic, MCMCMethod(:nuts))
result_vi = fit(prob_basic, UnifiedVIMethod())
```

# Performance Notes
Standardization is performed once during problem creation.
Pre-allocated temporaries are sized to the data dimensions.

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1.

See also: [`BDMLBasicProblem`](@ref), [`BDMLHierarchicalProblem`](@ref)
"""
function BDMLProblem(Y, D, X; model_type::Symbol = :basic)
    # Validate inputs
    n = length(Y)
    @assert length(D) == n "D must have same length as Y, got length(D)=$(length(D)) vs length(Y)=$n"
    @assert size(X, 1) == n "X must have n rows, got $(size(X, 1)) rows but n=$n"

    # Standardize data once
    Y_s, D_s, X_s, stats = standardize_data(Y, D, X)
    p = size(X_s, 2)

    # Pre-allocate temporaries
    μ_Y_cache = Vector{Float64}(undef, n)
    μ_D_cache = Vector{Float64}(undef, n)

    if model_type == :basic
        return BDMLBasicProblem(Y_s, D_s, X_s, stats, n, p, μ_Y_cache, μ_D_cache)
    elseif model_type == :hier
        return BDMLHierarchicalProblem(Y_s, D_s, X_s, stats, n, p, μ_Y_cache, μ_D_cache)
    else
        throw(ArgumentError("Unknown model_type: $model_type. Must be :basic or :hier"))
    end
end

# ==================== ACCESSOR FUNCTIONS ====================

"""
    nobs(prob::AbstractBDMLProblem)

Return the number of observations in the problem.

# Examples
```julia
prob = BDMLProblem(Y, D, X)
n = nobs(prob)  # Same as length(Y)
```
"""
nobs(prob::AbstractBDMLProblem) = prob.n

"""
    ncovariates(prob::AbstractBDMLProblem)

Return the number of control variables (covariates) in the problem.

# Examples
```julia
prob = BDMLProblem(Y, D, X)
p = ncovariates(prob)  # Same as size(X, 2)
```
"""
ncovariates(prob::AbstractBDMLProblem) = prob.p

"""
    model_type(prob::BDMLBasicProblem)
    model_type(prob::BDMLHierarchicalProblem)

Return the model type symbol (:basic or :hier).

Useful for dispatch and debugging.

# Examples
```julia
prob = BDMLProblem(Y, D, X; model_type=:basic)
model_type(prob)  # Returns :basic
```
"""
model_type(prob::BDMLBasicProblem) = :basic
model_type(prob::BDMLHierarchicalProblem) = :hier

"""
    standardization_stats(prob::AbstractBDMLProblem)

Return the standardization statistics for the problem.

Used to transform results back to original scale.

# Examples
```julia
prob = BDMLProblem(Y, D, X)
stats = standardization_stats(prob)
# Use stats.Y_sd, stats.D_sd, etc.
```
"""
standardization_stats(prob::AbstractBDMLProblem) = prob.stats

# ==================== CONVERSION ====================

"""
    BDMLProblem(data::BDMLData; model_type=:basic)

Create BDML problem from BDMLData struct.

Convenience constructor for working with the existing BDMLData type.

# Examples
```julia
data = BDMLData(Y, D, X)
prob = BDMLProblem(data; model_type=:basic)
```
"""
function BDMLProblem(data::BDMLData; model_type::Symbol = :basic)
    return BDMLProblem(data.Y, data.D, data.X; model_type)
end
