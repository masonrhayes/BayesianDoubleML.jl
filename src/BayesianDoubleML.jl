module BayesianDoubleML

using StatsAPI

# Core exports - dispatch-based API
export fit, BDMLProblem,
    AbstractBDMLProblem, BDMLBasicProblem, BDMLHierarchicalProblem,
    AbstractInferenceMethod, MCMCMethod, UnifiedVIMethod, SimpleVIMethod,
    MCMCNUTS, MCMCHMC, UnifiedVI, SimpleVI,
    # Variational families
    AbstractVariationalFamily, MeanField, LowRank, LowRankScore,
    MeanFieldVI, LowRankVI, LowRankScoreVI,
    # Method traits
    uses_sampling, supports_subsampling, is_deterministic, default_n_samples, default_n_iterations,
    # Accessors
    nobs, ncovariates, model_type, standardization_stats,
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
    # AD backends
    AutoReverseDiff, AutoForwardDiff, AutoZygote, AutoEnzyme, AutoMooncake,
    # DGP
    generate_dgp_table1

using Turing
using Turing.Variational
using AdvancedVI
using Bijectors
using ADTypes
using DynamicPPL
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

# Explicitly import AD backends for DifferentiationInterface compatibility
import Enzyme
import Mooncake

include("types.jl")
include("utils.jl")
include("model.jl")
include("alpha.jl")          # Basic extract_alpha for MCMC
include("alpha_extraction.jl")  # Additional extract_alpha methods for VI
include("coeftable.jl")    # StatsAPI-compliant coeftable with HPD intervals

# Multiple Dispatch System for BDML
# Provides unified fit() interface that dispatches on problem type and method type
include("problems.jl")      # Problem types: BDMLBasicProblem, BDMLHierarchicalProblem
include("methods.jl")       # Method types: MCMCMethod, UnifiedVIMethod, SimpleVIMethod
include("fit_dispatch.jl")  # Dispatch-based fit() functions

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
