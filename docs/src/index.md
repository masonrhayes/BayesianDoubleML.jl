# BayesianDoubleML.jl Documentation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://masonrhayes.github.io/BayesianDoubleML.jl/dev/)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Welcome to the documentation for **BayesianDoubleML.jl**, a Julia package for Bayesian inference in Double Machine Learning (DML) models.

## Overview

BayesianDoubleML.jl provides scalable and efficient Bayesian inference for causal effect estimation using the framework from [DiTraglia &amp; Liu (2025)](https://arxiv.org/abs/2508.12688). It offers both MCMC (as in the paper) as well as Variational Inference (VI) methods with multiple automatic differentiation backends.

## Key Features

- **Causal Inference**: Estimate treatment effects with uncertainty quantification
- **Multiple Inference Methods**:
  - MCMC with NUTS sampler
  - Variational Inference (Simple and Unified)
    - The `SimpleVIMethod()` relies on Turing.jl's VI implementation, offering simplicity and ease of use at the expense of less flexibility.
    - The `UnifiedVIMethod()` relies on `Bijectors.jl` and  `AdvancedVI.jl`, offering greater flexibility at the cost of slightly worse performance. This method supports both MeanFieldGuassian and LowRankGaussian [variational families](https://turinglang.org/AdvancedVI.jl/dev/families/) from `AdvancedVI.jl`.
- **Multiple AD Backends**: ReverseDiff, Mooncake, Zygote, ForwardDiff
  - Currently, Mooncake is only available with SimplifiedVIMethod, given some upstream compatibility issues
- **Automatic Subsampling**: For large datasets (n > 10,000)
- **Rich Diagnostics**: HPD intervals, p-values, ESS, Monte Carlo SE
- **StatsAPI Compliance**: Native `coeftable()` support

## The BDML Model

The Bayesian Double Machine Learning approach avoids regularization-induced confounding (RIC) via a bivariate reduced form parameterization:

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

# Generate synthetic data
n, p = 5000, 10
X = randn(n, p)
D = X[:, 1] + 0.5 * randn(n)  # Treatment
Y = 2.0 * D + X[:, 2] - 0.5 * X[:, 3] + randn(n)  # Outcome

# Create problem and fit
problem = BDMLProblem(Y, D, X; model_type=:hier)
result = fit(problem, MCMCMethod(:nuts); n_samples=1000, n_chains=4)

# View results
summary(result)
```

## Documentation Structure

- **[User Guide](user_guide.md)**: Detailed usage instructions with examples
- **[API Reference](api_reference.md)**: Complete function and type documentation

## Citation

If you use this package in your research, please cite the original paper:

> DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for Causal Inference". arXiv:2508.12688v1.

## License

MIT License
