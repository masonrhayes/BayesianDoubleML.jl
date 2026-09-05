module BayesianDoubleML

using StatsAPI

# Core exports - dispatch-based API with mutating fit!
export fit!, BDMLModel,
    AbstractBDMLModel, BDMLBasicModel, BDMLHierarchicalModel,
    AbstractInferenceMethod, MCMCMethod, VMPMethod,
    CollapsedVIMethod,
    MCMCNUTS, VMP, CollapsedVI,
    # VMP backends
    AbstractVMPBackend, RxInferVMP, ManualCoordinateAscentVMP,
    # Method traits
    uses_sampling, supports_subsampling, is_deterministic, default_n_samples, default_n_iterations,
    # Accessors
    nobs, ncovariates, model_type, standardization_stats, isfitted,
    # Results
    extract_alpha, BDMLData, AbstractBDMLResult, BDMLMCMCResult, BDMLVIResult, BDMLVMPResult,
    # Coeftable
    coeftable, BDMLCoeftable, confint, ess, pvalues, hpd_interval, mcse, rhat, rhat_statistic, chain_info,
    # StatsAPI functions
    coef, stderror, vcov,
    # Summary
    summary,
    # Utilities
    credible_interval,
    # AD backends for VI configuration
    AutoReverseDiff, AutoForwardDiff, AutoZygote, AutoMooncake,
    # DGP
    make_plr_DTL2025

using Turing
using AdvancedVI
using ADTypes
using DataFrames
using Distributions
using LinearAlgebra
using FlexiChains
using DifferentiationInterface
using Random
using ReverseDiff
using Statistics
using NaNMath
using Optimisers
using LogDensityProblems
using UnicodePlots
using Printf


include("types.jl")
include("utils.jl")
include("mcmc/mcmc_model.jl")          # MCMC model specifications (bdml_basic, bdml_hier)
include("alpha.jl")                   # Basic extract_alpha for MCMC, VI, VMP results

# Multiple Dispatch System for BDML
# Provides unified fit!() interface that dispatches on model type and method type
include("methods.jl")       # Method types: MCMCMethod, VMPMethod, CollapsedVIMethod
include("models.jl")      # Model types: BDMLBasicModel, BDMLHierarchicalModel
include("collapsed/collapsed.jl")
include("collapsed/collapsed_vi_model.jl")
include("collapsed/vi_diagnostics.jl")  # ELBO convergence checking (shared by CollapsedVI)
include("fit.jl")  # Dispatch-based fit!() functions
include("vmp/vmp_manual_coordinate_ascent.jl")
include("collapsed/collapsed_vi_fit.jl")

include("coeftable.jl")    # StatsAPI-compliant coeftable with HPD intervals

# Summary and visualization
include("summary.jl")

# Data Generating Processes (for simulations)
include("datasets/dgp.jl")
using .DGP

end
