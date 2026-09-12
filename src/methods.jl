# BDML Inference Method Types for Multiple Dispatch
# Defines the "how" - the algorithm/method for fitting problems

export AbstractInferenceMethod
export MCMCMethod, MCMCNUTS
export VMPMethod, VMP

# Abstract types

"""
    AbstractInferenceMethod

Abstract type for all inference methods/algorithms.

Subtypes define HOW to fit a BDML problem:
- MCMC methods: NUTS (reference method)
- Collapsed VI: ADVI on the analytically collapsed 3-d/5-d posterior
- VMP methods: conjugate message passing (manual or RxInfer backend)

Each concrete method type is dispatched on in the `fit()` function
along with the problem type to execute the appropriate algorithm.

See also: [`MCMCMethod`](@ref), [`CollapsedVIMethod`](@ref), [`VMPMethod`](@ref)
"""
abstract type AbstractInferenceMethod end

# MCMC methods

"""
    MCMCMethod <: AbstractInferenceMethod

MCMC (Markov Chain Monte Carlo) inference method.

Supports multiple MCMC algorithms via the `algorithm` field:
- :nuts - No-U-Turn Sampler (default, recommended)

# Fields
- `algorithm::Symbol`: Which MCMC algorithm (:nuts)
- `target_acceptance::Float64`: Target acceptance rate for NUTS (default: 0.8)
- `max_depth::Int`: Maximum tree depth for NUTS (default: 10)

# Constructors
```julia
MCMCMethod(:nuts; target_acceptance=0.8, max_depth=10)
MCMCNUTS(; target_acceptance=0.8, max_depth=10)  # Convenience
```

# Examples
```julia
# Default NUTS sampler
method = MCMCMethod(:nuts)

# Custom NUTS settings
method = MCMCMethod(:nuts; target_acceptance=0.9, max_depth=12)

# Convenience constructor
method = MCMCNUTS()
method = MCMCNUTS(; target_acceptance=0.9, max_depth=12)
```

# Notes
NUTS is the recommended MCMC algorithm. It automatically tunes
the trajectory length and is robust to step size choices.

See also: [`MCMCNUTS`](@ref)
"""
struct MCMCMethod <: AbstractInferenceMethod
    algorithm::Symbol
    target_acceptance::Float64
    max_depth::Int
    leapfrog_steps::Int
    step_size::Float64

    # Inner constructor with validation
    function MCMCMethod(
            algorithm::Symbol = :nuts;
            target_acceptance::Float64 = 0.8,
            max_depth::Int = 10,
            leapfrog_steps::Int = 10,
            step_size::Float64 = 0.1
        )
        return if algorithm == :nuts
            @assert 0 < target_acceptance < 1 "target_acceptance must be in (0, 1)"
            @assert max_depth > 0 "max_depth must be positive"
            new(:nuts, target_acceptance, max_depth, 0, 0.0)
        else
            throw(ArgumentError("Unknown MCMC algorithm: $algorithm. Use :nuts"))
        end
    end
end

"""
    MCMCNUTS(; target_acceptance=0.8, max_depth=10)

Convenience constructor for NUTS (No-U-Turn Sampler) MCMC method.

NUTS is the default and recommended MCMC algorithm. It automatically
tunes the trajectory length and is robust to step size choices.

# Arguments
- `target_acceptance::Float64=0.8`: Target acceptance rate (typical range: 0.6-0.9)
- `max_depth::Int=10`: Maximum tree depth (limits trajectory length)

# Examples
```julia
# Default NUTS
method = MCMCNUTS()

# Higher target acceptance for better exploration
method = MCMCNUTS(; target_acceptance=0.9)
```

See also: [`MCMCMethod`](@ref)
"""
MCMCNUTS(; target_acceptance::Float64 = 0.8, max_depth::Int = 10) =
    MCMCMethod(:nuts; target_acceptance, max_depth)

# VMP backend types

"""
    AbstractVMPBackend

Abstract type for Variational Message Passing backends.

Subtypes define the engine used for closed-form VMP updates:
- `RxInferVMP`: Uses the RxInfer.jl graph-based message passing (requires extension).
- `ManualCoordinateAscentVMP`: Direct vectorized conjugate updates without graph construction.
"""
abstract type AbstractVMPBackend end

"""
    RxInferVMP <: AbstractVMPBackend

VMP backend using RxInfer.jl (loaded via package extension).

# Constructor
```julia
RxInferVMP(; limit_stack_depth = nothing)
```

# Arguments
- `limit_stack_depth::Union{Nothing,Int}=nothing`: Optional recursion limit for
  large models. When `nothing`, RxInfer uses its default configuration.
"""
struct RxInferVMP <: AbstractVMPBackend
    limit_stack_depth::Union{Nothing, Int}
end
function RxInferVMP(; limit_stack_depth::Union{Nothing, Int} = nothing)
    limit_stack_depth === nothing || limit_stack_depth > 0 || throw(ArgumentError("limit_stack_depth must be positive"))
    return RxInferVMP(limit_stack_depth)
end

"""
    ManualCoordinateAscentVMP <: AbstractVMPBackend

VMP backend using closed-form manual coordinate ascent (sufficient-statistics form).

No AD, no graph construction, and memory use is O(p²) after preprocessing.

# Constructor
```julia
ManualCoordinateAscentVMP(; tolerance = 1.0e-8)
```

# Arguments
- `tolerance::Real=1.0e-8`: Relative parameter-change threshold for early stopping.
"""
struct ManualCoordinateAscentVMP <: AbstractVMPBackend
    tolerance::Float64
end
ManualCoordinateAscentVMP(; tolerance::Real = 1.0e-8) = ManualCoordinateAscentVMP(Float64(tolerance))

# VMP method

"""
    VMPMethod{B<:AbstractVMPBackend} <: AbstractInferenceMethod

Variational Message Passing (VMP) inference.

Uses a **conjugate reparameterization** of the BDML model so that all
variational message updates are available in closed form:

- ``\\Sigma \\sim \\text{InverseWishart}(\\nu_0, S_0)`` replaces the LKJ(4) +
  Half-Cauchy prior on the error covariance (DiTraglia & Liu 2025, Eq. 19).
- ``\\tau_\\delta, \\tau_\\gamma \\sim \\text{Gamma}(2, 1/2)`` on coefficient
  precisions (hierarchical variant; equivalent to ``\\text{InvGamma}(2,2)`` on
  the variances, preserving the paper's Student-``t(4)`` interpretation).

Reported uncertainty for ``\\alpha`` applies an effective residual
degrees-of-freedom correction to the mean-field covariance posterior. The raw
variational posterior remains available as `result.posterior.Σ_vmp`.

# Constructor
```julia
VMPMethod(;
    backend = ManualCoordinateAscentVMP(),
    ν0 = 4.0,
    S0 = nothing,
    aτ = 2.0,
    bτ = 0.5,
)
```

# Arguments
- `backend::AbstractVMPBackend`: VMP backend. Defaults to `ManualCoordinateAscentVMP()`.
- `ν0::Real=4.0`: Inverse-Wishart prior degrees of freedom (must exceed 3).
- `S0::Union{Nothing,AbstractMatrix}=nothing`: Inverse-Wishart scale matrix (2×2).
- `aτ::Real=2.0`, `bτ::Real=0.5`: Gamma hyperprior shape/scale on coefficient
  precisions (hierarchical model only).

# Examples
```julia
model = BDMLModel(df, :y, :d; model_type = :hier)
fit!(model, VMP(); n_iterations = 50)
```

To use the RxInfer backend instead, load the extension first:
```julia
using RxInfer
fit!(model, VMP(; backend = RxInferVMP()); n_iterations = 50)
```

See also: [`VMP`](@ref), [`RxInferVMP`](@ref), [`ManualCoordinateAscentVMP`](@ref)
"""
struct VMPMethod{B <: AbstractVMPBackend} <: AbstractInferenceMethod
    backend::B
    ν0::Float64
    S0::Union{Nothing, Matrix{Float64}}
    aτ::Float64
    bτ::Float64
end

function VMPMethod(;
        backend::AbstractVMPBackend = ManualCoordinateAscentVMP(),
        ν0::Real = 4.0,
        S0::Union{Nothing, AbstractMatrix} = nothing,
        aτ::Real = 2.0,
        bτ::Real = 0.5,
    )
    ν0 = Float64(ν0)
    aτ = Float64(aτ)
    bτ = Float64(bτ)
    ν0 > 3 || throw(ArgumentError("ν0 must exceed 3 so the 2×2 Inverse-Wishart prior has a finite mean"))
    aτ > 0 || throw(ArgumentError("aτ must be positive"))
    bτ > 0 || throw(ArgumentError("bτ must be positive"))
    if S0 !== nothing
        size(S0) == (2, 2) || throw(ArgumentError("S0 must be a 2×2 matrix"))
        issymmetric(S0) || throw(ArgumentError("S0 must be symmetric"))
        isposdef(Symmetric(S0)) || throw(ArgumentError("S0 must be positive definite"))
    end
    return VMPMethod(
        backend, ν0,
        S0 === nothing ? nothing : Matrix{Float64}(S0),
        aτ, bτ,
    )
end

"""
    VMP(; kwargs...)

Convenience constructor for `VMPMethod(; kwargs...)`.

Defaults to the `ManualCoordinateAscentVMP()` backend, which works without
any optional dependency. To use the `RxInferVMP()` backend instead, load
`RxInfer.jl` and pass `backend = RxInferVMP()`.

See [`VMPMethod`](@ref) for full documentation.
"""
VMP(; kwargs...) = VMPMethod(; kwargs...)

# Collapsed VI method

"""
    CollapsedVIMethod <: AbstractInferenceMethod

Collapsed variational inference integrates `(delta, gamma)` out analytically
and runs ADVI on the 3-dimensional (`:basic`) or 5-dimensional (`:hier`)
posterior for the error scales, correlation, and optional coefficient
variances. Conditional coefficient draws can then be recovered from their
exact joint Gaussian distribution.

The implementation uses the paper's priors and a
rank-aware SVD marginal likelihood. Each objective evaluation is `O(rank(X))`
and remains stable when `X` is rank deficient or `p` is close to `n`.

# Fields
- `ad_backend::Type{<:AbstractADType}`: AD backend (default `AutoReverseDiff`)
- `n_montecarlo::Int`: MC samples per ELBO gradient (default 10)
- `fullrank::Bool`: use a full-rank Gaussian if true, otherwise a mean-field Gaussian

# Examples
```julia
fit!(model, CollapsedVI(); n_iterations=1000, n_draws=2000)
fit!(model, CollapsedVI(; ad_backend=AutoMooncake, fullrank=true))
```

See also: [`VMPMethod`](@ref)
"""
struct CollapsedVIMethod <: AbstractInferenceMethod
    ad_backend::Type{<:AbstractADType}
    n_montecarlo::Int
    fullrank::Bool
    function CollapsedVIMethod(;
            ad_backend::Type{<:AbstractADType} = AutoReverseDiff,
            n_montecarlo::Int = 10,
            fullrank::Bool = true,
        )
        @assert n_montecarlo > 0 "n_montecarlo must be positive"
        return new(ad_backend, n_montecarlo, fullrank)
    end
end

"""
    CollapsedVI(; kwargs...)

Convenience constructor for `CollapsedVIMethod`.

See [`CollapsedVIMethod`](@ref).
"""
CollapsedVI(; kwargs...) = CollapsedVIMethod(; kwargs...)

# Trait functions

"""
    uses_sampling(method::AbstractInferenceMethod)

Return true if the method uses sampling (MCMC or Monte Carlo VI).

All current methods return true, but this enables future deterministic methods.
"""
uses_sampling(::MCMCMethod) = true
uses_sampling(::VMPMethod{<:AbstractVMPBackend}) = true
uses_sampling(::CollapsedVIMethod) = true

"""
    supports_subsampling(method::AbstractInferenceMethod)

Return true if the method supports mini-batch subsampling for large datasets.

No current method supports subsampling.
"""
supports_subsampling(::MCMCMethod) = false
supports_subsampling(::VMPMethod{<:AbstractVMPBackend}) = false
supports_subsampling(::CollapsedVIMethod) = false

"""
    is_deterministic(method::AbstractInferenceMethod)

Return true if the method is deterministic (no random sampling).

Currently all methods use sampling. Future deterministic methods (e.g., Laplace
approximation) would return true.
"""
is_deterministic(::AbstractInferenceMethod) = false

"""
    default_n_samples(method::AbstractInferenceMethod)

Return the default number of samples/draws for the method.

Returns 2000 for MCMC, 2000 for VI draw phase.
"""
default_n_samples(::MCMCMethod) = 2000
default_n_samples(::VMPMethod{<:AbstractVMPBackend}) = 2000
default_n_samples(::CollapsedVIMethod) = 2000

"""
    default_n_iterations(method::AbstractInferenceMethod)

Return the default number of optimization iterations for the method.

MCMC uses iterations as warm-up/tuning. VI uses iterations for optimization.
"""
default_n_iterations(::MCMCMethod) = 1000  # Warm-up iterations
default_n_iterations(::VMPMethod{<:AbstractVMPBackend}) = 50  # VMP converges in ~20-50 iterations
default_n_iterations(::CollapsedVIMethod) = 1000
