# BayesianDoubleML.jl Documentation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Welcome to the documentation for **BayesianDoubleML.jl**, a Julia package for Bayesian inference in Double Machine Learning (DML) models.

## Overview

BayesianDoubleML.jl provides scalable and efficient Bayesian inference for causal effect estimation using the framework from [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688). It offers both MCMC (as in the paper) as well as Variational Inference (VI) methods with multiple automatic differentiation backends.

## Key Features

- **Causal Inference**: Estimate treatment effects with uncertainty quantification
- **Multiple Inference Methods**:
  - MCMC with NUTS sampler
  - Variational Inference, with two primary methods:
    - The `SimpleVIMethod()` relies on Turing.jl's VI implementation, offering simplicity and ease of use at the expense of less flexibility.
    - The `UnifiedVIMethod()` relies on `Bijectors.jl` and  `AdvancedVI.jl`, offering greater flexibility. This method supports both MeanFieldGuassian and LowRankGaussian [variational families](https://turinglang.org/AdvancedVI.jl/dev/families/) from `AdvancedVI.jl`. However, this method does not currently support the Mooncake AD backend (which is extremely fast). Thus, for performance, it's recommended to stick with `SimpleVIMethod()` with `AutoMooncake` backend.
- **Multiple AD Backends**: ReverseDiff, Mooncake, Zygote, ForwardDiff.
- **Automatic Subsampling**: For large datasets (n > 10,000)

## The BDML Model

The Bayesian Double Machine Learning approach avoids regularization-induced confounding (RIC) via a bivariate reduced form parameterization.

As from the [DiTraglia and Liu (2025)](https://arxiv.org/abs/2508.12688) paper, the model takes the form of a bivariate reduced form regression model:

**Outcome equation:**

```math
Y = X'\delta + U \quad \text{(Equation 12)}
```

**Treatment equation:**

```math
D = X'\gamma + V \quad \text{(Equation 5)}
```

**Joint error distribution:**

```math
[U; V] \mid X \sim \mathcal{N}(0, \Sigma) \quad \text{(Equation 13)}
```

where ``\Sigma`` is the 2×2 covariance matrix:

```math
\Sigma = \begin{bmatrix} \sigma^2_U & \sigma_{UV} \\ \sigma_{UV} & \sigma^2_V \end{bmatrix}
```

**Causal effect recovery:**
The causal effect ``\alpha`` is recovered from the error covariance via:

```math
\alpha = \frac{\text{Cov}(U, V)}{\text{Var}(V)} = \frac{\sigma_{UV}}{\sigma^2_V} = \rho \frac{\sigma_U}{\sigma_V} \quad \text{(Equation 15)}
```

## Quick Start

```julia
using BayesianDoubleML
using StableRNGs

# Generate synthetic data
n = 200
p = 100
alpha_true = 2.0

rng = StableRNG(42)

# Generate data as DataFrame
df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)

# Create model and fit using DataFrame interface
# All columns except :y and :d are automatically used as covariates
model = BDMLModel(df, :y, :d; model_type=:hier)
fit!(model, MCMCMethod(:nuts); n_samples=1000, n_chains=4)

# View results
summary(model)

```

```julia-repl
# example results
╔══════════════════════════════════════════════════════════════════════╗
║         Bayesian Double ML Model Summary                   ║
╚══════════════════════════════════════════════════════════════════════╝
Model Information
───────────────────
  Model Type:       hier
  Standardization:  Y mean=0.238, sd=3.564; D mean=0.114, sd=1.490
Inference Method
──────────────────
  Method:           NUTS (No-U-Turn Sampler)
  Samples:          4000
MCMC Diagnostics
──────────────────
  Chains:           4
  Samples/Chain:    1000
  Total Samples:    4000
  ESS:              1779 (44.5% efficiency)
  ✓ Good effective sample size (ESS > 400)
  R-hat:            1.003
  ✓ Excellent convergence (R-hat < 1.01)
  MCSE:             0.0013
Causal Effect (α)
───────────────────
  Estimate:         1.9201
  Std Error:        0.1957
  95% CI:           [1.5402, 2.3027]
  95% HPD:          [1.5223, 2.2842]
──────────────────────────────────────────────────────────────

```

## Documentation Structure

- **[User Guide](user_guide.md)**: Detailed usage instructions with examples
- **[API Reference](api_reference.md)**: Complete function and type documentation

## Citation

If you use this package in your research, please consider citing the original paper (no affiliation):

> DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for Causal Inference". arXiv:2508.12688v1.

## License

MIT License
