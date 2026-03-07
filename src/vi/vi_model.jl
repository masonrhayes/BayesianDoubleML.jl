# Unified BDML Model for Variational Inference
# Implements LogDensityProblems interface with explicit Bijectors
# Supports all AD backends: ReverseDiff, Mooncake, Zygote, ForwardDiff

using LogDensityProblems
using LinearAlgebra
using AdvancedVI

export BDMLVIModel

"""
    BDMLVIModel{YType,DType,XType,T}

Unified model struct for all VI implementations.

Designed for efficiency and AD compatibility:
- Simple struct with concrete types (no abstract fields)
- Pre-allocated temporaries to avoid allocations during gradient computation
- Type-stable operations throughout
- No closures or dynamic dispatch

# Fields
- `Y::YType`: Outcome variable (Vector{Float64})
- `D::DType`: Treatment variable (Vector{Float64})
- `X::XType`: Control variables (Matrix{Float64})
- `n_data::Int`: Total number of observations (for likelihood scaling with subsampling)
- `model_type::Symbol`: :hier (hierarchical) or :basic
- `T::Type{T}`: Element type (Float64)
- `μ_Y_cache::Vector{T}`: Pre-allocated temporary for outcome mean
- `μ_D_cache::Vector{T}`: Pre-allocated temporary for treatment mean

# Example
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic, T=Float64)
```

# See Also
- `bijector(::BDMLVIModel)`: Bijector for unconstrained → constrained transformation
- `LogDensityProblems.logdensity(::BDMLVIModel, θ)`: Log-posterior computation
"""
struct BDMLVIModel{YType, DType, XType, T}
    Y::YType
    D::DType
    X::XType
    n_data::Int
    model_type::Symbol
    T::Type{T}
    # Pre-allocated temporaries for gradient computation (AD-safe)
    μ_Y_cache::Vector{T}
    μ_D_cache::Vector{T}
end

"""
    BDMLVIModel(Y, D, X; model_type=:hier, T=Float64)

Create a unified BDML model for variational inference.

Pre-allocates temporaries to avoid allocations during AD, improving performance
with all backends (ReverseDiff, Mooncake, Zygote, ForwardDiff).

# Arguments
- `Y::Vector{Float64}`: Outcome variable
- `D::Vector{Float64}`: Treatment variable  
- `X::Matrix{Float64}`: Control variables (covariates)

# Keyword Arguments
- `model_type::Symbol=:hier`: :hier (hierarchical) or :basic
- `T::Type=Float64`: Element type (default Float64)

# Returns
`BDMLVIModel`: Model instance ready for VI

# Examples
```julia
# Basic model
model = BDMLVIModel(Y, D, X; model_type=:basic)

# Hierarchical model (default)
model = BDMLVIModel(Y, D, X; model_type=:hier)
```

# Notes
The model automatically pre-allocates computation buffers sized to the data.
For subsampling, new temporaries are allocated sized to the batch.
"""
function BDMLVIModel(Y, D, X; model_type::Symbol = :hier, T::Type = Float64)
    @assert model_type in [:basic, :hier] "model_type must be :basic or :hier"

    n_data = length(Y)
    @assert n_data == length(D) "Y and D must have same length, got n_Y=$(length(Y)), n_D=$(length(D))"
    @assert n_data == size(X, 1) "X must have n_data rows, got $(size(X, 1)) rows but n_data=$n_data"

    # Pre-allocate temporaries for gradient computation
    # This avoids allocations during AD which improves performance
    μ_Y_cache = Vector{T}(undef, n_data)
    μ_D_cache = Vector{T}(undef, n_data)

    return BDMLVIModel(Y, D, X, n_data, model_type, T, μ_Y_cache, μ_D_cache)
end

"""
    LogDensityProblems.dimension(model::BDMLVIModel)

Return the number of parameters in the model.

Parameter counts:
- Basic: 2p (θ_Y, θ_D) + 2 (σ_U, σ_V) + 1 (ρ_raw) = 2p + 3
- Hierarchical: 2 (log_σ²_δ, log_σ²_γ) + 2p (θ_Y, θ_D) + 2 (σ_U, σ_V) + 1 (ρ_raw) = 2p + 5

where p = number of control variables (size(X, 2))

# Examples
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic)
d = LogDensityProblems.dimension(model)  # Returns 2*p + 3
```
"""
function LogDensityProblems.dimension(model::BDMLVIModel)
    n, p = size(model.X)
    if model.model_type == :hier
        return 2 * p + 5  # log_σ²_δ, log_σ²_γ, θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw
    else
        return 2 * p + 3  # θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw
    end
end

"""
    LogDensityProblems.capabilities(::Type{<:BDMLVIModel})

Declare that BDMLVIModel supports gradient computation (order 1).

This enables all first-order AD backends to work with the model.
"""
LogDensityProblems.capabilities(::Type{<:BDMLVIModel}) = LogDensityProblems.LogDensityOrder{1}()

"""
    unpack_parameters(model::BDMLVIModel, θ)

Unpack flat parameter vector θ into model components.

# Arguments
- `model::BDMLVIModel`: The model instance
- `θ::Vector{Float64}`: Flat parameter vector in constrained space

# Returns
For hierarchical models:
- `(θ_Y, θ_D, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ)`

For basic models:
- `(θ_Y, θ_D, σ_U, σ_V, ρ_raw)`

# Notes
Assumes parameters are already in constrained space (positive variances, ρ_raw in [0,1]).
This is called AFTER bijector transformation in logdensity().

# Examples
```julia
θ = rand(LogDensityProblems.dimension(model))
θ_Y, θ_D, σ_U, σ_V, ρ_raw = unpack_parameters(model, θ)
```
"""
function unpack_parameters(model::BDMLVIModel, θ)
    n, p = size(model.X)
    T = model.T

    if model.model_type == :hier
        # Hierarchical: [log_σ²_δ, log_σ²_γ, θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw]
        # After bijector: [σ²_δ, σ²_γ, θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
        log_σ²_δ = θ[1]
        log_σ²_γ = θ[2]
        θ_Y = θ[3:(2 + p)]
        θ_D = θ[(3 + p):(2 + 2p)]
        σ_U = θ[3 + 2p]
        σ_V = θ[4 + 2p]
        ρ_raw = θ[5 + 2p]

        # Transform log-variances to variances
        σ²_δ = exp(log_σ²_δ)
        σ²_γ = exp(log_σ²_γ)

        return θ_Y, θ_D, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ
    else
        # Basic: [θ_Y(p), θ_D(p), σ_U, σ_V, ρ_raw]
        θ_Y = θ[1:p]
        θ_D = θ[(p + 1):2p]
        σ_U = θ[2p + 1]
        σ_V = θ[2p + 2]
        ρ_raw = θ[2p + 3]

        return θ_Y, θ_D, σ_U, σ_V, ρ_raw
    end
end

"""
    AdvancedVI.subsample(model::BDMLVIModel, idx)

Create a subsampled model for mini-batch gradient estimation.

Returns a new model containing only observations at indices `idx`.
The original `n_data` is preserved for likelihood scaling.

# Arguments
- `model::BDMLVIModel`: The full model
- `idx::Vector{Int}`: Indices to include in the subsample

# Returns
`BDMLVIModel`: New model with subset of data and appropriately sized temporaries

# Examples
```julia
# Get a mini-batch of 100 observations
batch_idx = rand(1:n, 100)
batch_model = AdvancedVI.subsample(model, batch_idx)
```

# Notes
New temporaries are allocated sized to the batch (length(idx)).
"""
function AdvancedVI.subsample(model::BDMLVIModel, idx)
    n = length(idx)
    T = model.T

    # Create new pre-allocated temporaries for subsampled size
    μ_Y_cache = Vector{T}(undef, n)
    μ_D_cache = Vector{T}(undef, n)

    return BDMLVIModel(
        model.Y[idx],
        model.D[idx],
        model.X[idx, :],
        model.n_data,  # Keep original for likelihood scaling
        model.model_type,
        T,
        μ_Y_cache,
        μ_D_cache
    )
end
