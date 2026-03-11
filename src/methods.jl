# BDML Inference Method Types for Multiple Dispatch
# Defines the "how" - the algorithm/method for fitting problems

export AbstractInferenceMethod
export MCMCMethod, MCMCNUTS, MCMCHMC
export UnifiedVIMethod, SimpleVIMethod
export UnifiedVI, SimpleVI
export AbstractVariationalFamily, MeanField, LowRank, LowRankScore
export MeanFieldVI, LowRankVI, LowRankScoreVI

# Abstract types

"""
    AbstractInferenceMethod

Abstract type for all inference methods/algorithms.

Subtypes define HOW to fit a BDML problem:
- MCMC methods: NUTS, HMC, etc.
- VI methods: Unified (AdvancedVI), Simple (Turing vi())

Each concrete method type is dispatched on in the `fit()` function
along with the problem type to execute the appropriate algorithm.

See also: [`MCMCMethod`](@ref), [`UnifiedVIMethod`](@ref), [`SimpleVIMethod`](@ref)
"""
abstract type AbstractInferenceMethod end

# MCMC methods

"""
    MCMCMethod <: AbstractInferenceMethod

MCMC (Markov Chain Monte Carlo) inference method.

Supports multiple MCMC algorithms via the `algorithm` field:
- :nuts - No-U-Turn Sampler (default, recommended)
- :hmc - Hamiltonian Monte Carlo

# Fields
- `algorithm::Symbol`: Which MCMC algorithm (:nuts, :hmc)
- `target_acceptance::Float64`: Target acceptance rate for NUTS (default: 0.8)
- `max_depth::Int`: Maximum tree depth for NUTS (default: 10)
- `leapfrog_steps::Int`: Number of leapfrog steps for HMC (default: 10)
- `step_size::Float64`: Step size (ϵ) for HMC (default: 0.1)

# Constructors
```julia
MCMCMethod(:nuts; target_acceptance=0.8, max_depth=10)
MCMCMethod(:hmc; leapfrog_steps=10, step_size=0.1)
MCMCNUTS(; target_acceptance=0.8, max_depth=10)  # Convenience
MCMCHMC(; leapfrog_steps=10, step_size=0.1)     # Convenience
```

# Examples
```julia
# Default NUTS sampler
method = MCMCMethod(:nuts)

# Custom NUTS settings
method = MCMCMethod(:nuts; target_acceptance=0.9, max_depth=12)

# HMC sampler
method = MCMCMethod(:hmc; leapfrog_steps=20, step_size=0.05)

# Convenience constructors
method = MCMCNUTS()
method = MCMCHMC(; leapfrog_steps=15)
```

# Notes
NUTS is generally preferred over HMC as it automatically tunes
the trajectory length and is more robust to poor step size choices.
HMC can be faster for simple models with good step size tuning.

See also: [`MCMCNUTS`](@ref), [`MCMCHMC`](@ref)
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
        elseif algorithm == :hmc
            @assert leapfrog_steps > 0 "leapfrog_steps must be positive"
            @assert step_size > 0 "step_size must be positive"
            new(:hmc, 0.0, 0, leapfrog_steps, step_size)
        else
            throw(ArgumentError("Unknown MCMC algorithm: $algorithm. Use :nuts or :hmc"))
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

See also: [`MCMCMethod`](@ref), [`MCMCHMC`](@ref)
"""
MCMCNUTS(; target_acceptance::Float64 = 0.8, max_depth::Int = 10) =
    MCMCMethod(:nuts; target_acceptance, max_depth)

"""
    MCMCHMC(; leapfrog_steps=10, step_size=0.1)

Convenience constructor for HMC (Hamiltonian Monte Carlo) method.

HMC uses a fixed number of leapfrog steps. Can be faster than NUTS
for simple models but requires good step size tuning.

# Arguments
- `leapfrog_steps::Int=10`: Number of leapfrog integrator steps
- `step_size::Float64=0.1`: Step size (ϵ) for leapfrog integrator

# Examples
```julia
# Default HMC
method = MCMCHMC()

# More steps for better approximation
method = MCMCHMC(; leapfrog_steps=20, step_size=0.05)
```

See also: [`MCMCMethod`](@ref), [`MCMCNUTS`](@ref)
"""
MCMCHMC(; leapfrog_steps::Int = 10, step_size::Float64 = 0.1) =
    MCMCMethod(:hmc; leapfrog_steps, step_size)

# VI methods

# Variational family types

"""
    AbstractVariationalFamily

Abstract type for variational distribution families in UnifiedVI.

Subtypes define the shape of the variational posterior approximation:
- MeanField: Diagonal covariance (independent dimensions)
- LowRank: Low-rank plus diagonal covariance structure

Each family type is dispatched on for initialization in `initialize_variational_distribution`.
"""
abstract type AbstractVariationalFamily end

"""
    MeanField <: AbstractVariationalFamily

Mean-field (factorized) Gaussian variational family.

Uses a diagonal covariance matrix, assuming independence between
all parameters in the variational posterior.

Best for: High-dimensional problems, fast inference, when parameters
are approximately uncorrelated in the posterior.
"""
struct MeanField <: AbstractVariationalFamily end

"""
    LowRank <: AbstractVariationalFamily

Low-rank Gaussian variational family.

Uses a low-rank plus diagonal decomposition of the covariance:
Σ = D² + U*U' where U is d×r with r << d.

Best for: Problems with structured correlations, balancing between
mean-field and full-rank flexibility.

# Constructor
```julia
LowRank(rank::Int)  # rank is the number of low-rank factors
```
"""
struct LowRank <: AbstractVariationalFamily
    rank::Int

    function LowRank(rank::Int)
        @assert rank > 0 "LowRank rank must be positive"
        return new(rank)
    end
end

"""
    LowRankScore <: AbstractVariationalFamily

Low-rank Gaussian variational family using score gradient estimator (BBVI).

Uses a low-rank plus diagonal decomposition of the covariance:
Σ = D² + U*U' where U is d×r with r << d.

Uses the score gradient (REINFORCE) estimator with VarGrad control variate,
which can be more stable than the reparameterization gradient for some problems.

Best for: Problems where reparameterization gradient is unstable or when
exploring different gradient estimators.

# Constructor
```julia
LowRankScore(rank::Int)  # rank is the number of low-rank factors
```
"""
struct LowRankScore <: AbstractVariationalFamily
    rank::Int

    function LowRankScore(rank::Int)
        @assert rank > 0 "LowRankScore rank must be positive"
        return new(rank)
    end
end

"""
    UnifiedVIMethod{F<:AbstractVariationalFamily} <: AbstractInferenceMethod

Unified Variational Inference method using AdvancedVI with explicit Bijectors.

This is the primary VI implementation with:
- Explicit bijector transformations (unconstrained → constrained)
- Support for all AD backends (ReverseDiff, Mooncake, Zygote, ForwardDiff)
- Subsampling support for large datasets (n > 10,000)
- Multiple variational families (MeanField, LowRank) via type parameter
- Full control over the optimization process

# Type Parameters
- `F<:AbstractVariationalFamily`: The variational family type (MeanField, LowRank)

# Fields
- `ad_backend::Type`: AD backend (AutoReverseDiff, AutoMooncake, etc.)
- `subsample::Union{Bool, Nothing}`: Whether to use mini-batch gradients
- `batch_size::Int`: Mini-batch size (auto-computed if -1)
- `n_montecarlo::Int`: Number of Monte Carlo samples for gradient estimation (default: 10)
- `family::F`: Variational family instance (carries type-specific parameters like rank)

# Constructors
```julia
# Default MeanField
UnifiedVIMethod()

# LowRank with specific rank
UnifiedVIMethod(; family=LowRank(10))

# Convenience aliases
MeanFieldVI()      # Explicit mean-field
LowRankVI(10)      # Low-rank with rank 10
UnifiedVI()        # Backwards-compatible alias (defaults to MeanField)
```

# AD Backend Options
- `AutoReverseDiff` (default): Most stable, tape compilation, no warmup needed
- `AutoMooncake`: 5-10x faster after warmup, requires compilation
- `AutoZygote`: Source-to-source, higher memory usage
- `AutoForwardDiff`: Forward-mode, good for small p

# Subsampling
- `subsample=nothing`: Auto-enable for n >= 10,000
- `subsample=true`: Force subsampling with auto batch size
- `subsample=false`: Force full-batch
- `batch_size`: Manual control (default: min(256, ceil(n/1000)))

# Examples
```julia
# Default - MeanField with ReverseDiff
method = UnifiedVIMethod()

# Low-rank with rank 10
method = UnifiedVIMethod(; family=LowRank(10))

# With Mooncake (fast after warmup)
method = UnifiedVIMethod(; ad_backend=AutoMooncake, family=LowRank(10))

# Explicit subsampling control with low-rank
method = UnifiedVIMethod(; subsample=true, batch_size=512, family=LowRank(5))

# Convenience aliases
method = MeanFieldVI()           # Explicit mean-field
method = LowRankVI(10)           # Low-rank with rank 10
method = UnifiedVI()             # Backwards-compatible
```

# Notes
This implementation uses explicit bijectors (LogNormal, Beta) to transform
parameters from unconstrained space (where VI optimizes) to constrained space
(where model is defined). This provides better AD compatibility than relying
on Turing's automatic bijectors.

The variational family determines the covariance structure:
- MeanField: Diagonal only (fastest, assumes independence)
- LowRank: Diagonal + low-rank factors (balances speed and correlation capture)

See also: [`MeanField`](@ref), [`LowRank`](@ref), [`SimpleVIMethod`](@ref)
"""
struct UnifiedVIMethod{F <: AbstractVariationalFamily} <: AbstractInferenceMethod
    ad_backend::Type
    subsample::Union{Bool, Nothing}
    batch_size::Int
    n_montecarlo::Int
    family::F

    function UnifiedVIMethod(;
            ad_backend::Type = AutoReverseDiff,
            subsample::Union{Bool, Nothing} = nothing,
            batch_size::Int = -1,
            n_montecarlo::Int = 10,
            family::AbstractVariationalFamily = MeanField()
        )
        @assert n_montecarlo > 0 "n_montecarlo must be positive"
        return new{typeof(family)}(ad_backend, subsample, batch_size, n_montecarlo, family)
    end
end

"""
    SimpleVIMethod <: AbstractInferenceMethod

Simple Variational Inference method using Turing's native `vi()` function.

This is an alternative VI implementation that:
- Uses Turing's built-in ADVI with automatic bijectors
- Works well with AutoMooncake (5-10x speedup after warmup)
- Has simpler code, less maintenance overhead
- Does NOT support subsampling (Turing limitation)

Best for: Production use with Mooncake when n < 10,000 and you want
maximum performance after initial warmup.

# Fields
- `ad_backend::Type`: AD backend (AutoMooncake recommended, AutoReverseDiff works)

# Constructor
```julia
SimpleVIMethod(; ad_backend=AutoMooncake)
SimpleVI(; kwargs...)  # Convenience alias
```

# Examples
```julia
# Default - Mooncake (recommended for this implementation)
method = SimpleVIMethod()

# With ReverseDiff
method = SimpleVIMethod(; ad_backend=AutoReverseDiff)

# Convenience alias
method = SimpleVI()
```

# Comparison with UnifiedVIMethod
| Feature | UnifiedVIMethod | SimpleVIMethod |
|---------|----------------|----------------|
| Bijectors | Explicit | Automatic (Turing) |
| Subsampling | Yes | No |
| Mooncake | Has issues | Works well |
| ReverseDiff | Excellent | Good |
| Large data (n>10k) | Yes | No (memory) |
| Code complexity | More control | Simpler |

See also: [`UnifiedVIMethod`](@ref), [`SimpleVI`](@ref)
"""
struct SimpleVIMethod <: AbstractInferenceMethod
    ad_backend::Type

    function SimpleVIMethod(; ad_backend::Type = AutoMooncake)
        return new(ad_backend)
    end
end

"""
    UnifiedVI(; kwargs...)

Convenience alias for `UnifiedVIMethod()` with MeanField family.

Backwards-compatible - defaults to mean-field approximation.

See [`UnifiedVIMethod`](@ref) for full documentation.
"""
UnifiedVI(; kwargs...) = UnifiedVIMethod(; family = MeanField(), kwargs...)

"""
    MeanFieldVI(; kwargs...)

Convenience constructor for MeanField VI.

Creates a UnifiedVIMethod with MeanField variational family.

# Examples
```julia
method = MeanFieldVI()                      # Default MeanField
method = MeanFieldVI(; ad_backend=AutoMooncake)  # With Mooncake
```

See also: [`UnifiedVIMethod`](@ref), [`LowRankVI`](@ref)
"""
MeanFieldVI(; kwargs...) = UnifiedVIMethod(; family = MeanField(), kwargs...)

"""
    LowRankVI(rank::Int; kwargs...)

Convenience constructor for LowRank VI.

Creates a UnifiedVIMethod with LowRank variational family.

# Arguments
- `rank::Int`: Number of low-rank factors to use

# Examples
```julia
method = LowRankVI(10)                      # Low-rank with 10 factors
method = LowRankVI(5; ad_backend=AutoMooncake)   # With Mooncake
```

See also: [`UnifiedVIMethod`](@ref), [`MeanFieldVI`](@ref)
"""
LowRankVI(rank::Int; kwargs...) = UnifiedVIMethod(; family = LowRank(rank), kwargs...)

"""
    LowRankScoreVI(rank::Int; kwargs...)

Convenience constructor for LowRankScore VI (using score gradient).

Creates a UnifiedVIMethod with LowRankScore variational family.
Uses the score gradient (REINFORCE) estimator with VarGrad control variate.

# Arguments
- `rank::Int`: Number of low-rank factors to use

# Examples
```julia
method = LowRankScoreVI(10)                      # Low-rank score gradient with 10 factors
method = LowRankScoreVI(5; ad_backend=AutoMooncake)   # With Mooncake
```

See also: [`UnifiedVIMethod`](@ref), [`LowRankVI`](@ref)
"""
LowRankScoreVI(rank::Int; kwargs...) = UnifiedVIMethod(; family = LowRankScore(rank), kwargs...)

"""
    SimpleVI(; ad_backend=AutoMooncake)

Convenience alias for `SimpleVIMethod()`.

See [`SimpleVIMethod`](@ref) for full documentation.
"""
SimpleVI(; kwargs...) = SimpleVIMethod(; kwargs...)

# Trait functions

"""
    uses_sampling(method::AbstractInferenceMethod)

Return true if the method uses sampling (MCMC or Monte Carlo VI).

All current methods return true, but this enables future deterministic methods.
"""
uses_sampling(::MCMCMethod) = true
uses_sampling(::UnifiedVIMethod{<:AbstractVariationalFamily}) = true
uses_sampling(::SimpleVIMethod) = true

"""
    supports_subsampling(method::AbstractInferenceMethod)

Return true if the method supports mini-batch subsampling for large datasets.

Only `UnifiedVIMethod` supports subsampling. MCMC and SimpleVI do not.
"""
supports_subsampling(::MCMCMethod) = false
supports_subsampling(::UnifiedVIMethod{<:AbstractVariationalFamily}) = true
supports_subsampling(::SimpleVIMethod) = false

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
default_n_samples(::UnifiedVIMethod{<:AbstractVariationalFamily}) = 2000
default_n_samples(::SimpleVIMethod) = 2000

"""
    default_n_iterations(method::AbstractInferenceMethod)

Return the default number of optimization iterations for the method.

MCMC uses iterations as warm-up/tuning. VI uses iterations for optimization.
"""
default_n_iterations(::MCMCMethod) = 1000  # Warm-up iterations
default_n_iterations(::UnifiedVIMethod{<:AbstractVariationalFamily}) = 1000
default_n_iterations(::SimpleVIMethod) = 1000
