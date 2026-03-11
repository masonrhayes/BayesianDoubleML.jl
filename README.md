# BayesianDoubleML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Bayesian inference for Double Machine Learning (DML) with MCMC and Variational Inference.

This package implements [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688v1), Algorithm 1, using a bivariate reduced form parameterization to avoid regularization-induced confounding.

## Overview

BayesianDoubleML.jl provides scalable Bayesian causal inference with:

- **MCMC**: Exact inference with NUTS sampler (recommended for n < 1000)
- **VI**: Fast approximate inference with multiple AD backends (5-10+ times speedup)
- **Automatic subsampling**: For large datasets (n > 10,000)
- **StatsAPI compliance**: Native `coeftable()`, `coef()`, `stderror()` support

## Quick Start

```julia
using BayesianDoubleML
using StableRNGs

# Generate synthetic data
n = 200
p = 100
alpha_true = 2.0

rng = StableRNG(42)

# Generate data
Y, D, X = generate_dgp_table1(n, p, 2.0; alpha_true = alpha_true, rng = rng)

# Create model and fit
model = BDMLModel(Y, D, X; model_type = :hier)
fit!(model, MCMCMethod(:nuts); n_samples = 1000, n_chains = 4)


# Extract results
summary(model)
```

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/BayesianDoubleML.jl")
```

## Usage

### Creating Models

```julia
# Hierarchical model (recommended - adaptive shrinkage)
model = BDMLModel(Y, D, X; model_type=:hier)

# Basic model (fixed variance priors)
model = BDMLModel(Y, D, X; model_type=:basic)
```

### MCMC Inference

```julia
# NUTS sampler (default)
fit!(model, MCMCNUTS())

# Custom settings
fit!(model, MCMCNUTS(; target_acceptance=0.9); n_samples=2000, n_chains=4)
```

### VI Inference

```julia
# Default ReverseDiff
fit!(model, UnifiedVI())

# Mooncake (5-10x faster after warmup)
using Mooncake
fit!(model, SimpleVIMethod(; ad_backend = AutoMooncake))

```

### Results

```julia
# Extract causal effect
summary(model)

# MCMC diagnostics
ess(model)      # Effective sample size
rhat(model)     # Convergence (should be ≈ 1.0)
mcse(model)     # Monte Carlo SE

# VI convergence
model.result.elbo_history
model.result.converged
```

## API

### Model Types

- `BDMLModel(Y, D, X; model_type=:hier)` - Create model
- `BDMLBasicModel`, `BDMLHierarchicalModel` - Concrete types

### Methods

**MCMC:**

- `MCMCNUTS(; target_acceptance=0.8, max_depth=10)` - NUTS sampler
- `MCMCHMC(; leapfrog_steps=10, step_size=0.1)` - HMC sampler

**VI:**

- `UnifiedVI()` - AdvancedVI with bijectors
- `SimpleVI()` - Turing's native VI
- `MeanFieldVI()`, `LowRankVI(rank)` - Convenience constructors

### Core Functions

- `fit!(model, method; kwargs...)` - Fit model (mutating)
- `coeftable(model)` - Diagnostics table
- `extract_alpha(model)` - Causal effect samples
- `summary(model)` - Full summary

## Performance

**AD Backends:**

| Backend         | Speed        | Best For                     |
| --------------- | ------------ | ---------------------------- |
| AutoReverseDiff | Baseline     | Default, most stable         |
| AutoMooncake    | 5-10x faster | Production (requires warmup) |

**Warmup for Mooncake:**

```julia
# First run compiles (slow)
fit!(model, SimpleVI(); n_iterations=50)

# Subsequent runs are fast
fit!(model, SimpleVI(); n_iterations=1000, force=true)
```

## Model Variations

**BDML-Hier** (`:hier`, recommended):

- Hierarchical priors: σ²_δ, σ²_γ ~ InvGamma(2, 2)
- Adaptive shrinkage (~Student-t(4))
- Coverage: ~0.94

**BDML-Basic** (`:basic`):

- Fixed priors: δ, γ ~ N(0, 25·I)
- Simpler, uniform shrinkage
- Coverage: ~0.91-0.93

## Testing

```julia
using Pkg
Pkg.test("BayesianDoubleML")
```

## References

- DiTraglia & Liu (2025): [arXiv:2508.12688](https://arxiv.org/abs/2508.12688)
- Chernozhukov et al. (2018): [Econometrics Journal](https://doi.org/10.1111/ectj.12097)

## License

MIT License
