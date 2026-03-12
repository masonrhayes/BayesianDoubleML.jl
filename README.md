# BayesianDoubleML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Bayesian inference for Double Machine Learning with MCMC and Variational Inference.

Implements [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688), Algorithm 1, using a bivariate reduced form parameterization to avoid regularization-induced confounding.

## Features

- **MCMC**: NUTS sampler for inference using MCMC
- **VI**: Fast approximate inference with multiple AD backends
  - **Automatic subsampling with VI**: For large datasets (n > 10,000)
- **StatsAPI compliant**: `coeftable()`, `coef()`, `stderror()`, `vcov()`

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/BayesianDoubleML.jl")
```

## Quick Start

```julia
using BayesianDoubleML

# Generate synthetic data
Y, D, X, alpha_true, _ = generate_dgp_table1(200, 100, 2.0; alpha_true = 2.0)

# Create and fit model
model = BDMLModel(Y, D, X; model_type = :hier)
fit!(model, MCMCNUTS(); n_samples = 1000, n_chains = 4)

# View results
summary(model)
```

## Usage

### Model Types

```julia
# Hierarchical (recommended - adaptive shrinkage)
model = BDMLModel(Y, D, X; model_type = :hier)

# Basic (fixed variance priors)
model = BDMLModel(Y, D, X; model_type = :basic)
```

### Inference Methods

**MCMC:**

```julia
fit!(model, MCMCNUTS())  # Default NUTS
fit!(model, MCMCNUTS(; target_acceptance = 0.9); n_samples = 2000, n_chains = 4)
```

**VI:**

```julia
# UnifiedVI (default ReverseDiff)
fit!(model, UnifiedVI(); n_iterations = 1000)

# SimpleVI with Mooncake (faster after warmup)
using Mooncake
fit!(model, SimpleVI(; ad_backend = AutoMooncake))

# Low-rank variational family
fit!(model, LowRankVI(10))
```

### Results

```julia
# Summary statistics
summary(model)
coeftable(model)

# Extract causal effect samples
alpha_samples = extract_alpha(model)

# MCMC diagnostics
ess(model)   # Effective sample size
rhat(model)  # Convergence (should be ≈ 1.0)

# VI diagnostics
model.result.elbo_history
model.result.converged
```

## API Reference

### Core Functions

| Function                 | Description                        |
| ------------------------ | ---------------------------------- |
| `fit!(model, method)`  | Fit model (mutating)               |
| `coeftable(model)`     | Coefficient table with diagnostics |
| `summary(model)`       | Full summary                       |
| `extract_alpha(model)` | Causal effect samples              |

### Inference Methods

| Method              | Description               |
| ------------------- | ------------------------- |
| `MCMCNUTS()`      | NUTS sampler              |
| `UnifiedVI()`     | AdvancedVI with bijectors |
| `SimpleVI()`      | Turing's native VI        |
| `MeanFieldVI()`   | Mean-field VI             |
| `LowRankVI(rank)` | Low-rank VI               |

### StatsAPI Functions

`coef(model)`, `stderror(model)`, `vcov(model)`, `confint(model; level=0.95)`

## Performance

| AD Backend      | Speed        | Best For                         |
| --------------- | ------------ | -------------------------------- |
| AutoReverseDiff | Baseline     | Default, most stable             |
| AutoMooncake    | 5-10x faster | Fast inference (requires warmup) |

## Model Variations

**Hierarchical** (`:hier`, recommended):

- Hierarchical priors with adaptive shrinkage
- Coverage: ~94%

**Basic** (`:basic`):

- Fixed variance priors
- Coverage: ~91-93%

## References

- DiTraglia & Liu (2025): [arXiv:2508.12688](https://arxiv.org/abs/2508.12688)
- Chernozhukov et al. (2018): [Econometrics Journal](https://doi.org/10.1111/ectj.12097)

## License

MIT License
