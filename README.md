# BayesianDoubleML.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](<https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826>)](https://github.com/SciML/SciMLStyle)

Bayesian Double Machine Learning with various inference methods:

- Markov chain Monte Carlo (MCMC), as in the original paper [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688), as well as:
- Alternative approaches using variational inference:
  - CollapsedVI: using a collapsed (i.e., Rao-Blackwellized) version of the model and Automatic Differentiation Variational Inference, and
  - VMP (Variational Message Passing) based methods

This package implement BDML as in  [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688), Algorithm 1, using a bivariate reduced form parameterization to avoid regularization-induced confounding.

In addition, an experimental version of the Bayes-DR model from [Antonelli et al (2022)](https://doi.org/10.1111/biom.13417) (see below).

## Features

- **MCMC**: NUTS sampler for inference using MCMC
- **CollapsedVI**: Fast approximate inference on the analytically collapsed (Rao-Blackwellized) posterior, with multiple AD backends
- **VMP**: Reparameterisation of the problem using conjugate-exponential family for extremely fast inference, with manual implementation (default) or, optionally, an [RxInfer.jl](https://rxinfer.com/) backend; `α` uncertainty includes an effective residual-DF correction
- **StatsAPI compliant**: `coeftable()`, `coef()`, `stderror()`, `vcov()`
- **Experimental Bayes-DR**: Binary-treatment ATE and continuous-treatment exposure-response estimation following [Antonelli et al (2022)](https://doi.org/10.1111/biom.13417)

## Installation

BayesianDoubleML.jl requires Julia 1.12 or later.

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/BayesianDoubleML.jl")
```

For development, the package, tests, documentation, and examples form a Pkg
workspace backed by the root `Manifest.toml`. Instantiate all workspace members
with:

```julia
using Pkg
Pkg.instantiate(; workspace = true)
```

Activate `test`, `docs`, or `examples` before changing that member's direct
dependencies; Pkg will update the shared root manifest.

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

**CollapsedVI:**

```julia
# CollapsedVI (default ReverseDiff)
fit!(model, CollapsedVI(); n_iterations = 1000)

# CollapsedVI with Mooncake, mean-field family
using Mooncake
fit!(model, CollapsedVI(; ad_backend = AutoMooncake, fullrank = false))
```

**VMP (default: manual backend, optional RxInfer backend):**

```julia
# Manual backend (default, no extra dependency)
fit!(model, VMP(); n_iterations = 50)

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

Reported `α` uncertainty applies an effective residual degrees-of-freedom
correction to the mean-field covariance posterior. In `result.posterior`, `Σ`
is calibrated for `α`, `Σ_vmp` is the raw variational posterior, and
`effective_df` records the correction (`summary` reports `Effective DF`).

You can also run the manual coordinate-ascent backend explicitly without RxInfer:

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

### Experimental Bayes-DR

The experimental Bayes-DR model combines separate Bayesian treatment and Gaussian outcome nuisance models with a doubly robust estimator for binary-treatment ATEs and continuous-treatment exposure-response curves. Its standard error includes the paper's empirical-bootstrap and posterior nuisance-parameter variance components.

```julia
model = BayesDRModel(Y, T, X)  # T must be coded as 0/1
fit!(
    model,
    BayesDRMCMC();
    n_samples = 1000,
    n_burn = 500,
    n_chains = 2,
    n_boot = 500,
)

coef(model)
stderror(model)
confint(model)
```

Reproduce the paper's primary simulation design with:

```julia
df = make_irm_APD2022(100, 500; scenario = :nonlinear)
model = BayesDRModel(df, :y, :d)
```

This initial implementation supports continuous outcomes and linear additive nuisance models. It reports a posterior-corrected confidence interval, not a posterior credible interval.

## API Reference

### Core Functions

| Function                | Description                        |
| ----------------------- | ---------------------------------- |
| `fit!(model, method)` | Fit model (mutating)               |
| `coeftable(model)`    | Coefficient table with diagnostics |
| `summary(model)`      | Full summary                       |

### Inference Methods

| Method                          | Description                              |
| ------------------------------- | ---------------------------------------- |
| `MCMCNUTS()`                  | NUTS sampler                             |
| `CollapsedVI()`               | Collapsed ADVI (full-rank or mean-field) |
| `VMP()`                       | Conjugate VMP (default: manual backend)  |
| `ManualCoordinateAscentVMP()` | Manual VMP backend (no extension)        |
| `RxInferVMP()`                | RxInfer VMP backend                      |
| `BayesDRMCMC()`               | Experimental binary-ATE and continuous exposure-response Bayes-DR |

### StatsAPI Functions

`coef(model)`, `stderror(model)`, `vcov(model)`, `confint(model; level=0.95)`

## Performance

| Inference Method              | Best For                                                   |
| ----------------------------- | ---------------------------------------------------------- |
| VMP (ManualCoordinateAscent)  | Fastest (<seconds), with good approximation of posterior when p < n |
| CollapsedVI (AutoReverseDiff) | Quite fast, good approximation of posterior                |
| CollapsedVI (AutoMooncake)    | Fast, ~5-10x faster than CollapsedVI with AutoReverseDiff |
| MCMC                          | Most accurate inference                                    |

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
- Antonelli, Papadogeorgou, & Dominici (2022): [Biometrics](https://doi.org/10.1111/biom.13417)

## License

MIT License
