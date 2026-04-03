# BDML Model Types for Multiple Dispatch
# Defines the "what" - the model specification with pre-computed data

export AbstractBDMLModel, BDMLBasicModel, BDMLHierarchicalModel, BDMLModel
export nobs, ncovariates, model_type, standardization_stats, isfitted

"""
    AbstractBDMLModel

Abstract type for all BDML model specifications.

Model types encapsulate:
- Data (Y, D, X) - already standardized
- Standardization statistics (for transforming back)
- Pre-allocated temporaries (for performance)
- Metadata (n, p, model_type)
- Fitting results and state (is_fitted, result, last_method)

Each concrete subtype represents a different model specification
(basic vs hierarchical) and can be fitted with different methods
(MCMC, VI unified, VI simple) via multiple dispatch.

After fitting with `fit!()`, results are stored in the model and can be
extracted using `coeftable()`, `extract_alpha()`, `summary()`, etc.
"""
abstract type AbstractBDMLModel end

"""
    BDMLBasicModel <: AbstractBDMLModel

Basic BDML model specification implementing BDML-Basic from 
DiTraglia & Liu (2025), Algorithm 1.

# Fields
- `Y::Vector{Float64}`: Standardized outcome variable
- `D::Vector{Float64}`: Standardized treatment variable  
- `X::Matrix{Float64}`: Standardized control variables (covariates)
- `stats::StandardizationStats`: Statistics for transforming back to original scale
- `n::Int`: Number of observations
- `p::Int`: Number of control variables
- `result::Union{Nothing, AbstractBDMLResult}`: Stores fitting results after `fit!()`
- `is_fitted::Bool`: Whether the model has been fitted
- `last_method::Union{Nothing, AbstractInferenceMethod}`: Method used in last fit

# Model Specification (Section 4, Equations 12-13)

**Reduced form:**
    ``Y = X'\\delta + U``          (Eq. 12)
    ``D = X'\\gamma + V``          (Eq. 5)

**Priors:**
- ``\\delta \\sim N(0, 25\\cdot I_p)``      [Outcome reduced form coefficients]
- ``\\gamma \\sim N(0, 25\\cdot I_p)``      [Treatment reduced form coefficients]  
- ``\\sigma_U \\sim \\text{Cauchy}^+(0, 2.5)`` [Outcome error scale]
- ``\\sigma_V \\sim \\text{Cauchy}^+(0, 2.5)`` [Treatment error scale]
- ``R \\sim \\text{LKJ}(4)``            [Correlation matrix]

For VI, uses ``\\text{Beta}(2,2)`` correlation parameterization instead of LKJCholesky.

**Causal effect recovery:**
    ``\\alpha = \\rho\\cdot\\sigma_U / \\sigma_V``       (Eq. 15)

This is the BDML-Basic variation with fixed prior variances (Section 6, Table 1).

See also: [`BDMLHierarchicalModel`](@ref), [`BDMLModel`](@ref)

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.
"""
mutable struct BDMLBasicModel <: AbstractBDMLModel
    Y::Vector{Float64}
    D::Vector{Float64}
    X::Matrix{Float64}
    stats::StandardizationStats
    n::Int
    p::Int
    # Result storage
    result::Union{Nothing, AbstractBDMLResult}
    is_fitted::Bool
    last_method::Union{Nothing, AbstractInferenceMethod}
end

"""
    BDMLHierarchicalModel <: AbstractBDMLModel

Hierarchical BDML model specification implementing BDML-Hier from 
DiTraglia & Liu (2025), Algorithm 1.

# Fields
- `Y::Vector{Float64}`: Standardized outcome variable
- `D::Vector{Float64}`: Standardized treatment variable  
- `X::Matrix{Float64}`: Standardized control variables (covariates)
- `stats::StandardizationStats`: Standardization statistics
- `n::Int`: Number of observations
- `p::Int`: Number of control variables
- `result::Union{Nothing, AbstractBDMLResult}`: Stores fitting results after `fit!()`
- `is_fitted::Bool`: Whether the model has been fitted
- `last_method::Union{Nothing, AbstractInferenceMethod}`: Method used in last fit

# Model Specification (Section 4, Equations 12-13)

**Reduced form:**
    ``Y = X'\\delta + U``          (Eq. 12)
    ``D = X'\\gamma + V``          (Eq. 5)

**Hierarchical priors:**
- ``\\sigma^2_\\delta \\sim \\text{InvGamma}(2, 2)``       [Variance hyperparameter for ``\\delta``]
- ``\\sigma^2_\\gamma \\sim \\text{InvGamma}(2, 2)``       [Variance hyperparameter for ``\\gamma``]
- ``\\delta \\sim N(0, \\sigma^2_\\delta\\cdot I_p)``         [Outcome coefficients with adaptive shrinkage]
- ``\\gamma \\sim N(0, \\sigma^2_\\gamma\\cdot I_p)``         [Treatment coefficients with adaptive shrinkage]
- ``\\sigma_U \\sim \\text{Cauchy}^+(0, 2.5)``       [Outcome error scale]
- ``\\sigma_V \\sim \\text{Cauchy}^+(0, 2.5)``       [Treatment error scale]
- ``R \\sim \\text{LKJ}(4)``                 [Correlation matrix]

For VI, uses ``\\text{Beta}(2,2)`` correlation parameterization.

**Causal effect recovery:**
    ``\\alpha = \\rho\\cdot\\sigma_U / \\sigma_V``            (Eq. 15)

**Hierarchical structure:**
This is equivalent to placing independent ``\\text{Student-}t(4)`` distributions on each 
coefficient marginally. The ``\\text{InvGamma}(2, 2)`` hyperprior provides adaptive 
shrinkage that learns the appropriate regularization from data (Section 6, Table 1).

This can improve performance when p is large relative to n, and the paper's
simulations show BDML-Hier achieves better coverage (0.94) than BDML-Basic (0.91-0.93).

See also: [`BDMLBasicModel`](@ref), [`BDMLModel`](@ref)

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.
"""
mutable struct BDMLHierarchicalModel <: AbstractBDMLModel
    Y::Vector{Float64}
    D::Vector{Float64}
    X::Matrix{Float64}
    stats::StandardizationStats
    n::Int
    p::Int
    # Result storage
    result::Union{Nothing, AbstractBDMLResult}
    is_fitted::Bool
    last_method::Union{Nothing, AbstractInferenceMethod}
end

"""
    BDMLModel(Y, D, X; model_type=:basic)

Factory function to create appropriate BDML model type for Algorithm 1
from DiTraglia & Liu (2025).

Standardizes data once during model creation.
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
`AbstractBDMLModel`: Either `BDMLBasicModel` or `BDMLHierarchicalModel`

# Algorithm 1 Variations

**BDML-Basic** (`model_type=:basic`):
- Places independent ``N(0, 25\\cdot I)`` priors on ``\\delta`` and ``\\gamma``
- Fixed shrinkage across all covariates
- Recommended for interpretability and simplicity

**BDML-Hier** (`model_type=:hier`):
- Places hierarchical ``\\text{InvGamma}(2, 2)`` priors on ``\\sigma^2_\\delta`` and ``\\sigma^2_\\gamma``
- Adaptive shrinkage equivalent to ``\\text{Student-}t(4)`` on coefficients
- Better coverage in simulations (Table 1: 0.94 vs 0.91-0.93)
- Recommended as default choice

Both variations recover ``\\alpha`` via Equation 15: ``\\alpha = \\rho\\cdot\\sigma_U / \\sigma_V``

# Examples
```julia
# Create basic model
model_basic = BDMLModel(Y, D, X; model_type=:basic)

# Create hierarchical model  
model_hier = BDMLModel(Y, D, X; model_type=:hier)

# Fit with different methods
fit!(model_basic, MCMCMethod(:nuts))
fit!(model_hier, UnifiedVIMethod())
```

# Performance Notes
Standardization is performed once during model creation.
Results are stored in the model after calling `fit!()`.

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1.

See also: [`BDMLBasicModel`](@ref), [`BDMLHierarchicalModel`](@ref), [`fit!`](@ref)
"""
function BDMLModel(Y, D, X; model_type::Symbol = :basic)
    # Validate inputs
    n = length(Y)
    @assert length(D) == n "D must have same length as Y, got length(D)=$(length(D)) vs length(Y)=$n"
    @assert size(X, 1) == n "X must have n rows, got $(size(X, 1)) rows but n=$n"

    # Standardize data once
    Y_s, D_s, X_s, stats = standardize_data(Y, D, X)
    p = size(X_s, 2)

    # Initialize result storage
    result = nothing
    is_fitted = false
    last_method = nothing

    if model_type == :basic
        return BDMLBasicModel(
            Y_s, D_s, X_s, stats, n, p,
            result, is_fitted, last_method
        )
    elseif model_type == :hier
        return BDMLHierarchicalModel(
            Y_s, D_s, X_s, stats, n, p,
            result, is_fitted, last_method
        )
    else
        throw(ArgumentError("Unknown model_type: $model_type. Must be :basic or :hier"))
    end
end

# Accessor functions

"""
    nobs(model::AbstractBDMLModel)

Return the number of observations in the model.

# Examples
```julia
model = BDMLModel(Y, D, X)
n = nobs(model)  # Same as length(Y)
```
"""
nobs(model::AbstractBDMLModel) = model.n

"""
    ncovariates(model::AbstractBDMLModel)

Return the number of control variables (covariates) in the model.

# Examples
```julia
model = BDMLModel(Y, D, X)
p = ncovariates(model)  # Same as size(X, 2)
```
"""
ncovariates(model::AbstractBDMLModel) = model.p

"""
    model_type(model::BDMLBasicModel)
    model_type(model::BDMLHierarchicalModel)

Return the model type symbol (:basic or :hier).

Useful for dispatch and debugging.

# Examples
```julia
model = BDMLModel(Y, D, X; model_type=:basic)
model_type(model)  # Returns :basic
```
"""
model_type(model::BDMLBasicModel) = :basic
model_type(model::BDMLHierarchicalModel) = :hier

"""
    standardization_stats(model::AbstractBDMLModel)

Return the standardization statistics for the model.

Used to transform results back to original scale.

# Examples
```julia
model = BDMLModel(Y, D, X)
stats = standardization_stats(model)
# Use stats.Y_sd, stats.D_sd, etc.
```
"""
standardization_stats(model::AbstractBDMLModel) = model.stats

"""
    isfitted(model::AbstractBDMLModel)

Return `true` if the model has been fitted (i.e., `fit!()` has been called).

# Examples
```julia
model = BDMLModel(Y, D, X)
isfitted(model)  # Returns false

fit!(model)
isfitted(model)  # Returns true
```
"""
isfitted(model::AbstractBDMLModel) = model.is_fitted

"""
    BDMLModel(data::BDMLData; model_type=:basic)

Create BDML model from BDMLData struct.

Convenience constructor for working with the existing BDMLData type.

# Examples
```julia
data = BDMLData(Y, D, X)
model = BDMLModel(data; model_type=:basic)
```
"""
function BDMLModel(data::BDMLData; model_type::Symbol = :basic)
    return BDMLModel(data.Y, data.D, data.X; model_type)
end

# Pretty printing for models
function Base.show(io::IO, model::AbstractBDMLModel)
    model_type_str = model isa BDMLBasicModel ? "Basic" : "Hierarchical"
    fitted_str = isfitted(model) ? "fitted" : "not fitted"
    println(io, "BDML$(model_type_str)Model ($(fitted_str))")
    println(io, "  Observations: $(nobs(model))")
    println(io, "  Covariates: $(ncovariates(model))")
    return if isfitted(model)
        println(io, "  Result: $(typeof(model.result))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::AbstractBDMLModel)
    return show(io, model)
end

"""
    BDMLModel(df::DataFrame, y::Symbol, d::Symbol; model_type::Symbol=:basic, x_cols=nothing)

Create a BDML model from a DataFrame, specifying the outcome and treatment columns.

# Arguments
- `df::DataFrame`: The data as a DataFrame
- `y::Symbol`: Column name for the outcome variable
- `d::Symbol`: Column name for the treatment variable

# Keyword Arguments
- `model_type::Symbol=:basic`: 
  - `:basic` for BDML-Basic (fixed prior variances)
  - `:hier` for BDML-Hier (adaptive hierarchical priors)
- `x_cols`: Column names for covariates. If `nothing` (default), uses all columns except `y` and `d`

# Returns
`AbstractBDMLModel`: Either `BDMLBasicModel` or `BDMLHierarchicalModel`

# Examples
```julia
using DataFrames

# Generate data
df = make_plr_DTL2025(200, 100, 2.0)

# Default: use all columns except :y and :d as covariates
model = BDMLModel(df, :y, :d; model_type=:hier)

# Explicit covariate selection
model = BDMLModel(df, :y, :d; model_type=:hier, x_cols=[:X1, :X2, :X3])

# Using column indices or regex
model = BDMLModel(df, :y, :d; model_type=:hier, x_cols=names(df, r"^X"))
```

See also: [`BDMLModel`](@ref), [`BDMLBasicModel`](@ref), [`BDMLHierarchicalModel`](@ref)
"""
function BDMLModel(df::DataFrame, y::Symbol, d::Symbol; model_type::Symbol = :basic, x_cols = nothing)
    # Extract Y and D
    Y = df[:, y]
    D = df[:, d]

    # Determine X columns
    if x_cols === nothing
        # Default: all columns except y and d
        all_cols = Symbol.(names(df))
        x_cols_sym = filter(col -> col != y && col != d, all_cols)
    else
        x_cols_sym = Symbol.(x_cols)
    end

    # Extract X as Matrix
    X = Matrix(df[:, x_cols_sym])

    # Create the model using the standard constructor
    return BDMLModel(Y, D, X; model_type)
end
