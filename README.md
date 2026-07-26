# BayesianDoubleML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](<https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826>)](https://github.com/SciML/SciMLStyle)

Bayesian Double Machine Learning with various inference methods:

- Markov chain Monte Carlo (MCMC),
- Automatic Differentiation Variational Inference (ADVI), and
- Variational Message Passing (VMP)

This package implements BDML as in  [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688), Algorithm 1, using a bivariate reduced form parameterization to avoid regularization-induced confounding.

## Features

- **MCMC**: NUTS sampler for inference using MCMC
- **VI**: Fast approximate inference with multiple AD backends
  - **Automatic subsampling with VI**: For large datasets (n > 10,000)
- **VMP**: Reparameterisation of the problem using conjugate-exponential family for extremely fast inference, with manual implementation or, optionally, an [RxInfer.jl](https://rxinfer.com/) backend
- **StatsAPI compliant**: `coeftable()`, `coef()`, `stderror()`, `vcov()`

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/BayesianDoubleML.jl")
```

## Quick Start

```julia
using BayesianDoubleML

# Generate synthetic data as DataFrame
df = make_plr_DTL2025(200, 100, 2.0; alpha = 2.0)

# Create and fit model - all columns except :y and :d are covariates
model = BDMLModel(df, :y, :d; model_type = :hier)
fit!(model, MCMCNUTS(); n_samples = 1000, n_chains = 4)

# View results
summary(model)
```

## Usage

### Model Types

```julia
# Hierarchical (recommended - adaptive shrinkage)
# Pass DataFrame with outcome column :y and treatment column :d
model = BDMLModel(df, :y, :d; model_type = :hier)

# Basic (fixed variance priors)
model = BDMLModel(df, :y, :d; model_type = :basic)

# Or use the Y, D, X interface
model = BDMLModel(Y, D, X; model_type = :hier)
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

**VMP with the optional RxInfer extension:**

```julia
using Pkg
Pkg.add("RxInfer")

using RxInfer  # activates BayesianDoubleMLRxInferExt
fit!(model, VMP(; backend = RxInferVMP()); n_iterations = 50)
```

VMP replaces the LKJ + Half-Cauchy covariance prior with the conjugate
`InverseWishart(ν₀, S₀)` prior used in the paper's theoretical specification
(Equation 19). The fixed and hierarchical coefficient priors retain their
existing interpretations. Configure the covariance prior in the `VMP()`
constructor:

```julia
method = VMP(;
    backend = RxInferVMP(),
    ν0 = 4.0,
    S0 = nothing,
)
fit!(model, method; n_iterations = 50)
```

You can also run the manual coordinate-ascent backend without RxInfer:

```julia
method = VMP(; backend = ManualCoordinateAscentVMP())
fit!(model, method; n_iterations = 50)
```

### Results

```julia
# Summary statistics
summary(model)
coeftable(model)
```

## API Reference

### Core Functions

| Function                | Description                        |
| ----------------------- | ---------------------------------- |
| `fit!(model, method)` | Fit model (mutating)               |
| `coeftable(model)`    | Coefficient table with diagnostics |
| `summary(model)`      | Full summary                       |

### Inference Methods

| Method                          | Description                             |
| ------------------------------- | --------------------------------------- |
| `MCMCNUTS()`                  | NUTS sampler                            |
| `UnifiedVI()`                 | AdvancedVI with bijectors               |
| `SimpleVI()`                  | Turing's native VI                      |
| `MeanFieldVI()`               | Mean-field VI (AdvancedVI)              |
| `LowRankVI(rank)`             | Low-rank VI (AdvancedVI)                |
| `VMP()`                       | Conjugate VMP (default: manual backend) |
| `ManualCoordinateAscentVMP()` | Manual VMP backend (no extension)       |
| `RxInferVMP()`                | RxInfer VMP backend                     |

### StatsAPI Functions

`coef(model)`, `stderror(model)`, `vcov(model)`, `confint(model; level=0.95)`

## Performance

| Inference Method             | Best For                                                 |
| ---------------------------- | -------------------------------------------------------- |
| VMP (ManualCoordinateAscent) | Fastest (<seconds), with good appoximation of posterior |
| ADVI (AutoReverseDiff)       | Quite fast, good appoximation of posterior               |
| ADVI (AutoMooncake)          | Fast, ~5-10x faster than ADVI with AutoReverseDiff      |
| MCMC                         | Most accurate inference                                  |

## Model Variations

**Hierarchical** (`:hier`, recommended):

- Hierarchical priors with adaptive shrinkage
- Coverage: ~94% (from paper)

**Basic** (`:basic`):

- Fixed variance priors
- Coverage: ~91-93% (from paper)

## References

- DiTraglia & Liu (2025): [arXiv:2508.12688](https://arxiv.org/abs/2508.12688)
- Chernozhukov et al. (2018): [Econometrics Journal](https://doi.org/10.1111/ectj.12097)

## License

MIT License
