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

# Model Specification

Implements the bivariate reduced form model from DiTraglia & Liu (2025), 
Equations 13-15:

**Reduced form equations:**
    ``Y = X'\\delta + U``          (Eq. 12)
    ``D = X'\\gamma + V``          (Eq. 5)

**Joint error distribution:**
    ``[U; V] | X \\sim N(0, \\Sigma)``  (Eq. 13)

where the causal effect ``\\alpha`` is recovered via:
    ``\\alpha = \\text{Cov}(U, V) / \\text{Var}(V) = \\rho\\cdot\\sigma_U / \\sigma_V``   (Eq. 15)

# Fields
- `Y::YType`: Outcome variable (Vector{Float64})
- `D::DType`: Treatment variable (Vector{Float64})
- `X::XType`: Control variables (Matrix{Float64})
- `n_data::Int`: Total number of observations (for likelihood scaling with subsampling)
- `model_type::Symbol`: :hier (hierarchical) or :basic
- `T::Type{T}`: Element type (Float64)
- `μ_Y_cache::Vector{T}`: Pre-allocated temporary for outcome mean ``(X'\\delta)``
- `μ_D_cache::Vector{T}`: Pre-allocated temporary for treatment mean ``(X'\\gamma)``

# Example
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic, T=Float64)
```

# See Also
- `bijector(::BDMLVIModel)`: Bijector for unconstrained → constrained transformation
- `LogDensityProblems.logdensity(::BDMLVIModel, θ)`: Log-posterior computation
- `unpack_parameters`: Extract δ, γ, and other parameters from flat vector

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4.
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

Parameter counts (paper notation ``\\delta`` for outcome, ``\\gamma`` for treatment):
- Basic: ``2p`` (``\\delta``, ``\\gamma``) + 2 (``\\sigma_U``, ``\\sigma_V``) + 1 (``\\rho_\\text{raw}``) = ``2p + 3``
- Hierarchical: 2 (``\\log_\\sigma^2_\\delta``, ``\\log_\\sigma^2_\\gamma``) + ``2p`` (``\\delta``, ``\\gamma``) + 2 (``\\sigma_U``, ``\\sigma_V``) + 1 (``\\rho_\\text{raw}``) = ``2p + 5``

where ``p`` = number of control variables (size(X, 2))

See DiTraglia & Liu (2025), Section 4, Equations 12-13.

# Examples
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic)
d = LogDensityProblems.dimension(model)  # Returns 2*p + 3
```
"""
function LogDensityProblems.dimension(model::BDMLVIModel)
    n, p = size(model.X)
    if model.model_type == :hier
        return 2 * p + 5  # log_σ²_δ, log_σ²_γ, δ(p), γ(p), σ_U, σ_V, ρ_raw
    else
        return 2 * p + 3  # δ(p), γ(p), σ_U, σ_V, ρ_raw
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

Unpack flat parameter vector θ into model components using paper notation.

# Arguments
- `model::BDMLVIModel`: The model instance
- `θ::Vector{Float64}`: Flat parameter vector in constrained space

# Returns
For hierarchical models:
- `(δ, γ, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ)`

For basic models:
- `(δ, γ, σ_U, σ_V, ρ_raw)`

where:
- ``\\delta``: Reduced form coefficients for Y on X ``(Eq. 12)``
- ``\\gamma``: Reduced form coefficients for D on X ``(Eq. 5)``
- ``\\sigma_U``: Outcome error standard deviation
- ``\\sigma_V``: Treatment error standard deviation  
- ``\\rho_\\text{raw}``: Correlation parameter in [0,1] (transforms to ``\\rho = 2\\rho_\\text{raw} - 1``)
- ``\\sigma^2_\\delta``: Hierarchical variance hyperparameter for ``\\delta``
- ``\\sigma^2_\\gamma``: Hierarchical variance hyperparameter for ``\\gamma``

# Notes
Assumes parameters are already in constrained space (positive variances, ρ_raw in [0,1]).
This is called AFTER bijector transformation in logdensity().

See DiTraglia & Liu (2025), Section 4, Equations 12-13.

# Examples
```julia
θ = rand(LogDensityProblems.dimension(model))
δ, γ, σ_U, σ_V, ρ_raw = unpack_parameters(model, θ)
```
"""
function unpack_parameters(model::BDMLVIModel, θ)
    n, p = size(model.X)
    T = model.T

    if model.model_type == :hier
        # Hierarchical: [log_σ²_δ, log_σ²_γ, δ(p), γ(p), σ_U, σ_V, ρ_raw]
        # After bijector: [σ²_δ, σ²_γ, δ..., γ..., σ_U, σ_V, ρ_raw]
        log_σ²_δ = θ[1]
        log_σ²_γ = θ[2]
        δ = θ[3:(2 + p)]
        γ = θ[(3 + p):(2 + 2p)]
        σ_U = θ[3 + 2p]
        σ_V = θ[4 + 2p]
        ρ_raw = θ[5 + 2p]

        # Transform log-variances to variances
        σ²_δ = exp(log_σ²_δ)
        σ²_γ = exp(log_σ²_γ)

        return δ, γ, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ
    else
        # Basic: [δ(p), γ(p), σ_U, σ_V, ρ_raw]
        δ = θ[1:p]
        γ = θ[(p + 1):2p]
        σ_U = θ[2p + 1]
        σ_V = θ[2p + 2]
        ρ_raw = θ[2p + 3]

        return δ, γ, σ_U, σ_V, ρ_raw
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
The reduced form means X'δ and X'γ are computed on the subsample.
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
