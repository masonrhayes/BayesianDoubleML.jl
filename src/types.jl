# Abstract parent type for all BDML results
abstract type AbstractBDMLResult end

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

struct StandardizationStats
    Y_mean::Float64
    Y_sd::Float64
    D_mean::Float64
    D_sd::Float64
    X_mean::Vector{Float64}
    X_sd::Vector{Float64}
end

struct BDMLMCMCResult <: AbstractBDMLResult
    chain::MCMCChains.Chains
    alpha_samples::Vector{Float64}
    alpha_samples_standardized::Vector{Float64}
    std_stats::StandardizationStats
    model_type::Symbol
end

# Show methods are defined in coeftable.jl

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
