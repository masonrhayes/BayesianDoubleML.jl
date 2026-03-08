"""
    AbstractBDMLResult

Abstract parent type for all Bayesian Double Machine Learning results.

Subtypes include:
- `BDMLMCMCResult`: Results from MCMC inference
- `BDMLVIResult`: Results from Variational Inference

All subtypes implement the result interface with methods like:
- `extract_alpha()`: Extract α samples
- `coeftable()`: Generate coefficient table
- `confint()`: Compute credible intervals
"""
abstract type AbstractBDMLResult end

"""
    BDMLData

Container for BDML input data with standardized types.

# Fields
- `Y::Vector{Float64}`: Outcome variable
- `D::Vector{Float64}`: Treatment variable
- `X::Matrix{Float64}`: Control covariates (n×p)
- `n::Int`: Number of observations
- `p::Int`: Number of covariates

# Constructor
```julia
BDMLData(Y, D, X)
```

Automatically converts inputs to Float64 and validates dimensions.
"""
struct BDMLData
    Y::Vector{Float64}
    D::Vector{Float64}
    X::Matrix{Float64}
    n::Int
    p::Int
end

function BDMLData(Y, D, X)
    n = length(Y)
    p = size(X, 2)
    @assert length(D) == n "D must have same length as Y"
    @assert size(X, 1) == n "X must have n rows"
    return BDMLData(Vector{Float64}(Y), Vector{Float64}(D), Matrix{Float64}(X), n, p)
end

"""
    StandardizationStats

Statistics from data standardization, used for back-transformation.

# Fields
- `Y_mean::Float64`: Mean of outcome before standardization
- `Y_sd::Float64`: Std dev of outcome before standardization
- `D_mean::Float64`: Mean of treatment before standardization
- `D_sd::Float64`: Std dev of treatment before standardization
- `X_mean::Vector{Float64}`: Means of covariates before standardization
- `X_sd::Vector{Float64}`: Std devs of covariates before standardization

Used internally to transform results back to original scale after fitting
on standardized data.
"""
struct StandardizationStats
    Y_mean::Float64
    Y_sd::Float64
    D_mean::Float64
    D_sd::Float64
    X_mean::Vector{Float64}
    X_sd::Vector{Float64}
end

"""
    BDMLMCMCResult <: AbstractBDMLResult

Results from MCMC inference using NUTS or HMC sampling.

# Fields
- `chain::MCMCChains.Chains`: Full MCMC chain from Turing
- `alpha_samples::Vector{Float64}`: Causal effect samples (original scale)
- `alpha_samples_standardized::Vector{Float64}`: Causal effect samples (standardized scale)
- `std_stats::StandardizationStats`: Statistics for back-transformation
- `model_type::Symbol`: :basic or :hier

# Usage
```julia
result = fit(problem, MCMCNUTS())
alpha_mean = mean(result.alpha_samples)
ci = credible_interval(result)
```

See also: [`BDMLVIResult`](@ref), [`AbstractBDMLResult`](@ref)
"""
struct BDMLMCMCResult <: AbstractBDMLResult
    chain::MCMCChains.Chains
    alpha_samples::Vector{Float64}
    alpha_samples_standardized::Vector{Float64}
    std_stats::StandardizationStats
    model_type::Symbol
end

# Show methods are defined in coeftable.jl

"""
    BDMLVIResult <: AbstractBDMLResult

Results from Variational Inference (ADVI) approximation.

# Fields
- `variational_posterior::Any`: Variational distribution from AdvancedVI
- `alpha_samples::Vector{Float64}`: Causal effect samples (original scale)
- `alpha_samples_standardized::Vector{Float64}`: Causal effect samples (standardized scale)
- `std_stats::StandardizationStats`: Statistics for back-transformation
- `model_type::Symbol`: :basic or :hier
- `variational_family::Symbol`: :meanfield or :lowrank
- `n_iterations::Int`: Number of optimization iterations performed
- `elbo_history::Vector{Float64}`: ELBO values during optimization
- `converged::Bool`: Whether convergence criteria were met
- `final_elbo::Float64`: Final ELBO value

# Usage
```julia
result = fit(problem, UnifiedVI())
alpha_mean = mean(result.alpha_samples)
println("Converged: ", result.converged)
println("Final ELBO: ", result.final_elbo)
```

See also: [`BDMLMCMCResult`](@ref), [`AbstractBDMLResult`](@ref)
"""
struct BDMLVIResult <: AbstractBDMLResult
    variational_posterior::Any  # Bijectors.TransformedDistribution or similar
    alpha_samples::Vector{Float64}
    alpha_samples_standardized::Vector{Float64}
    std_stats::StandardizationStats
    model_type::Symbol
    variational_family::Symbol  # :fullrank or :meanfield
    n_iterations::Int
    elbo_history::Vector{Float64}  # For convergence monitoring
    converged::Bool
    final_elbo::Float64
end

# Show methods are defined in coeftable.jl
