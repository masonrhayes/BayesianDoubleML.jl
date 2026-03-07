# BayesianDoubleML.jl

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A Julia package for Bayesian inference in Double Machine Learning (DML) models using both MCMC and Variational Inference (VI).

## Overview

BayesianDoubleML.jl provides scalable and efficient Bayesian inference for causal effect estimation using the Double Machine Learning (Chernozhukov et al., 2018) framework. The package supports both exact MCMC inference and fast approximate VI, with multiple backend options including AutoReverseDiff, AutoMooncake, AutoZygote, and AutoForwardDiff.

### Key Features

- **Causal Inference**: Estimate treatment effects α with uncertainty quantification
- **Multiple Dispatch Interface**: Unified `fit(problem, method)` API for all algorithms
- **MCMC Samplers**: NUTS and HMC with convergence diagnostics
- **Variational Inference**: 
  - Unified VI (AdvancedVI with explicit bijectors) - supports subsampling for large datasets
  - Simple VI (Turing's native ADVI) - optimized for production use
- **Multiple AD Backends**: ReverseDiff, Mooncake, Zygote, ForwardDiff
- **Automatic Subsampling**: Mini-batch gradient estimation for datasets with n > 10,000
- **Rich Diagnostics**: HPD intervals, p-values, effective sample size (ESS), Monte Carlo SE
- **StatsAPI Compliance**: Native `coeftable()` support with formatted coefficient tables

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("BayesianDoubleML")
```

### Basic Usage

```julia
using BayesianDoubleML

# Generate or load your data
n, p = 5000, 10  # 5000 observations, 10 covariates
X = randn(n, p)
D = X[:, 1] + 0.5 * randn(n)  # Treatment (confounded with X)
Y = 2.0 * D + X[:, 2] - 0.5 * X[:, 3] + randn(n)  # Outcome

# Create a problem
problem = BDMLProblem(Y, D, X; model_type=:hier)

# Fit with MCMC (NUTS sampler - default)
result_mcmc = fit(problem, MCMCMethod(:nuts); n_samples=2000, n_chains=4)

# Or fit with VI (fast approximate inference)
result_vi = fit(problem, UnifiedVIMethod(); n_iterations=1000)

# Extract and analyze results
mean(result_mcmc.alpha_samples)  # Average causal effect
std(result_mcmc.alpha_samples)     # Uncertainty

# Generate coefficient table with HPD intervals, p-values, ESS
coeftable_result = coeftable(result_mcmc)
println(coeftable_result)
```

## Usage Examples

### 1. Creating Problems

```julia
# Basic model (fixed variance priors)
problem_basic = BDMLProblem(Y, D, X; model_type=:basic)

# Hierarchical model (adaptive shrinkage)
problem_hier = BDMLProblem(Y, D, X; model_type=:hier)
```

### 2. MCMC Methods

```julia
# Default NUTS sampler (recommended)
result = fit(problem, MCMCMethod(:nuts))

# Custom NUTS settings
result = fit(problem, MCMCMethod(:nuts; target_acceptance=0.9, max_depth=12))

# HMC sampler (requires more tuning)
result = fit(problem, MCMCMethod(:hmc; leapfrog_steps=20, step_size=0.05))

# Convenience constructors
result = fit(problem, MCMCNUTS())
result = fit(problem, MCMCHMC(; leapfrog_steps=15))
```

### 3. VI Methods

```julia
# Unified VI with default AutoReverseDiff
result = fit(problem, UnifiedVIMethod())

# Unified VI with Mooncake (5-10x faster after warmup)
# Note: Run once to compile rules, then subsequent calls are fast
result = fit(problem, UnifiedVIMethod(; ad_backend=AutoMooncake))

# Unified VI with subsampling for large datasets (auto-enabled when n >= 10000)
result = fit(problem, UnifiedVIMethod(; subsample=true, batch_size=512))

# Simple VI (Turing's native implementation, optimized for Mooncake)
result = fit(problem, SimpleVIMethod())
result = fit(problem, SimpleVIMethod(; ad_backend=AutoReverseDiff))

# Convenience constructors
result = fit(problem, UnifiedVI())
result = fit(problem, SimpleVI())
```

### 4. Accessing Results

```julia
# All result types
result.alpha_samples              # Causal effect on original scale
result.alpha_samples_standardized # Causal effect on standardized scale

# MCMC results
result.chain                      # MCMCChains.Chains object

# VI results
result.variational_posterior      # Variational distribution
result.elbo_history               # Convergence monitoring
result.converged                  # Boolean convergence flag
result.final_elbo                 # Final ELBO value

# Diagnostics (both MCMC and VI)
ct = coeftable(result)
# Shows: estimate, std error, MCSE, ESS, HPD interval, p-value
```

### 5. Diagnostics and Utilities

```julia
# HPD credible intervals (shortest intervals)
hpd_interval(result.alpha_samples; level=0.95)

# Standard credible intervals
confint(result; level=0.95)

# Effective sample size (ESS)
ess(result)

# P-values
effective_sample_size(result.alpha_samples)  # For MCMC
pvalues(result)

# Monte Carlo Standard Error (MCSE)
mcse(result.alpha_samples)
```

## API Reference

### Problem Types

- `BDMLProblem(Y, D, X; model_type=:basic)` - Create problem from data
- `BDMLBasicProblem` - Basic model with fixed variance priors
- `BDMLHierarchicalProblem` - Hierarchical model with adaptive shrinkage

### Method Types

**MCMC:**
- `MCMCMethod(algorithm; kwargs...)` - Generic MCMC method
- `MCMCNUTS(; target_acceptance=0.8, max_depth=10)` - NUTS sampler
- `MCMCHMC(; leapfrog_steps=10, step_size=0.1)` - HMC sampler

**VI:**
- `UnifiedVIMethod(; ad_backend=AutoReverseDiff, subsample=nothing, batch_size=-1)` - AdvancedVI
- `SimpleVIMethod(; ad_backend=AutoMooncake)` - Turing's native VI
- `UnifiedVI()` - Convenience alias
- `SimpleVI()` - Convenience alias

### Core Functions

- `fit(problem, method; kwargs...)` - Fit model with specified method
- `coeftable(result)` - Generate coefficient table with diagnostics
- `extract_alpha(result)` - Extract α samples from result
- `credible_interval(samples; level=0.95)` - Compute credible intervals

### Diagnostics

- `hpd_interval(samples; level=0.95)` - Highest Posterior Density intervals
- `effective_sample_size(samples)` - ESS using autocorrelation
- `mcse(samples)` - Monte Carlo Standard Error
- `pvalues(result)` - Two-sided p-values
- `ess(result)` - Effective sample size accessor
- `confint(result; level=0.95)` - Confidence/credible intervals

## Project Structure

```
src/
├── BayesianDoubleML.jl    # Main module, exports
├── types.jl              # Result types (BDMLResult, BDMLVIResult)
├── utils.jl              # Standardization, helper functions
├── problems.jl           # Problem types (BDMLBasicProblem, BDMLHierarchicalProblem)
├── methods.jl            # Method types (MCMCMethod, UnifiedVIMethod, SimpleVIMethod)
├── fit_dispatch.jl       # Multiple dispatch fit() implementations
├── model.jl              # Core Turing models (bdml_basic, bdml_hier)
├── alpha_extraction.jl   # Alpha extraction from samples
├── alpha.jl              # Basic extract_alpha for MCMC
├── coeftable.jl          # StatsAPI-compliant coefficient tables
├── fit.jl                # Legacy fitting functions
├── vi/                   # Unified VI implementation
│   ├── vi_model.jl       # BDMLVIModel (LogDensityProblems interface)
│   ├── vi_bijectors.jl   # Bijectors for unconstrained→constrained
│   ├── vi_logdensity.jl  # Log-posterior with Jacobian adjustment
│   └── vi_fit.jl         # Core VI fitting
├── vi_simple/            # Simple VI (Turing's native vi())
│   ├── model_vi.jl       # VI-compatible Turing models
│   └── fit_vi.jl
├── zygote_vi/            # Zygote-specific VI (deprecated)
├── enzyme_vi/            # Enzyme-specific VI (deprecated)
├── model_vi_subsampled.jl # Legacy subsampled VI
└── fit_vi_subsampled.jl   # Legacy subsampled fitting

archive/
└── experimental/         # Archived experimental VI methods
    └── vi_methods.jl     # FullRankVI and LowRankVI (archived due to issues)

docs/
└── STATUS_UPDATE.md      # Development status and progress

test/                     # Test suite (to be implemented)
```

## Performance Tips

### Choosing AD Backends

- **AutoReverseDiff** (Default): Most stable, works with all methods. Enable `compile=true` for repeated evaluations.
- **AutoMooncake**: 5-10x faster after warmup. First 1-2 runs compile differentiation rules (~50-100 iterations). Best for production/batch processing.
- **AutoZygote**: Source-to-source AD. Higher memory usage but works well for experimentation.
- **AutoForwardDiff**: Forward-mode dual numbers. Best for small models (p < 20), constant compilation time.

### Large Datasets (n > 10,000)

Use UnifiedVIMethod with subsampling (auto-enabled):
```julia
result = fit(problem, UnifiedVIMethod())  # Auto-subsamples when n >= 10000
```

### Warmup for Mooncake

```julia
# First run: compile rules (slow)
result_warmup = fit(problem, UnifiedVIMethod(; ad_backend=AutoMooncake); 
                  n_iterations=50)

# Subsequent runs: 5-10x faster
result = fit(problem, UnifiedVIMethod(; ad_backend=AutoMooncake); 
            n_iterations=1000)
```

## Troubleshooting

### VI Methods

- **UnifiedVIMethod** with Mooncake: Use AutoReverseDiff if you encounter SubArray compatibility issues
- **SimpleVIMethod**: Does NOT support subsampling (Turing limitation). Use UnifiedVIMethod for large datasets.

### Memory Issues

For very large datasets, the MCMC chains can consume significant memory. Use VI methods for memory-efficient approximate inference.

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.

## Contributing

Contributions are welcome! Please see the development guidelines:
- Follow the SciML Style Guide
- Use 4 spaces for indentation
- Maintain type stability
- Add tests for new features
- Document all public functions with docstrings

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This package builds on the Julia probabilistic programming ecosystem:
- Turing.jl for the PPL framework
- AdvancedVI.jl for variational inference algorithms
- LogDensityProblems.jl for the log density interface
- MCMCChains.jl for chain diagnostics
- Bijectors.jl for parameter transformations
