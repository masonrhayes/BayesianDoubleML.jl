module BayesianDoubleML

using StatsAPI

# Core exports - dispatch-based API with mutating fit!
export fit!, BDMLModel,
    AbstractBDMLModel, BDMLBasicModel, BDMLHierarchicalModel,
    AbstractInferenceMethod, MCMCMethod, UnifiedVIMethod, SimpleVIMethod,
    MCMCNUTS, UnifiedVI, SimpleVI,
    # Variational families
    AbstractVariationalFamily, MeanField, LowRank, LowRankScore,
    MeanFieldVI, LowRankVI, LowRankScoreVI,
    # Method traits
    uses_sampling, supports_subsampling, is_deterministic, default_n_samples, default_n_iterations,
    # Accessors
    nobs, ncovariates, model_type, standardization_stats, isfitted,
    # Results
    extract_alpha, BDMLData, AbstractBDMLResult, BDMLMCMCResult, BDMLVIResult,
    # Coeftable
    coeftable, BDMLCoeftable, confint, ess, pvalues, hpd_interval, mcse, rhat, rhat_statistic, chain_info,
    # StatsAPI functions
    coef, stderror, vcov,
    # Summary
    summary,
    # Utilities
    credible_interval, check_convergence,
    # AD backends for VI configuration
    AutoReverseDiff, AutoForwardDiff, AutoZygote, AutoMooncake,
    # DGP
    make_plr_DTL2025

using Turing
using Turing.Variational
using AdvancedVI
using Bijectors
using ADTypes
using DataFrames
using Distributions
using LinearAlgebra
using MCMCChains
using Random
using ReverseDiff
using Statistics
using NaNMath
using Optim
using Optimisers
using LogDensityProblems
using LogDensityProblemsAD
using UnicodePlots
using Printf


include("types.jl")
include("utils.jl")
include("mcmc/mcmc_model.jl")          # MCMC model specifications (bdml_basic, bdml_hier)
include("alpha.jl")                   # Basic extract_alpha for MCMC
include("alpha_extraction.jl")       # Additional extract_alpha methods for VI

# Multiple Dispatch System for BDML
# Provides unified fit!() interface that dispatches on model type and method type
include("methods.jl")       # Method types: MCMCMethod, UnifiedVIMethod, SimpleVIMethod
include("models.jl")      # Model types: BDMLBasicModel, BDMLHierarchicalModel
include("fit.jl")  # Dispatch-based fit!() functions

include("coeftable.jl")    # StatsAPI-compliant coeftable with HPD intervals

# VI model definitions (used by dispatch system)
include("vi/vi_model.jl")
include("vi/vi_bijectors.jl")
include("vi/vi_logdensity.jl")
include("vi/vi_diagnostics.jl")  # ELBO convergence checking
include("vi/vi_fit.jl")

# Simple VI model definitions (used by dispatch system)
include("vi_simple/model_vi.jl")
include("vi_simple/fit_vi.jl")

# Summary and visualization
include("summary.jl")

# Data Generating Processes (for simulations)
include("datasets/dgp.jl")
using .DGP

end
