# User Guide

This guide demonstrates how to use BayesianDoubleML.jl for causal inference with different inference methods.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Inference Methods](#inference-methods)
  - [MCMC (NUTS)](#mcmc-nuts)
  - [Simple VI with Mooncake](#simple-vi-with-mooncake)
  - [Unified VI with AutoReverseDiff](#unified-vi-with-autoreversediff)
- [Understanding Results](#understanding-results)
- [Performance Tips](#performance-tips)

## Installation

```julia
using Pkg
Pkg.add("BayesianDoubleML")
```

## Basic Usage

### Creating a Problem

All analyses start by creating a `BDMLProblem` from your data:

```julia
using BayesianDoubleML

# Your data
Y = your_outcome_vector  # Vector{Float64}
D = your_treatment_vector  # Vector{Float64}  
X = your_covariate_matrix  # Matrix{Float64}

# Create problem with hierarchical model (recommended)
problem = BDMLProblem(Y, D, X; model_type=:hier)

# Or use basic model with fixed priors
problem = BDMLProblem(Y, D, X; model_type=:basic)
```

### Model Types

**Hierarchical Model (`:hier`) - RECOMMENDED:**
- Uses adaptive shrinkage via hierarchical priors
- Equivalent to Student-t(4) priors on coefficients
- Better coverage in simulations (0.94 vs 0.91-0.93)

**Basic Model (`:basic`):**
- Uses fixed N(0, 25·I) priors
- Simpler interpretation
- Good for baseline comparisons

## Inference Methods

### MCMC (NUTS)

MCMC using the No-U-Turn Sampler provides exact posterior inference and is recommended for small-to-medium datasets (n < 1000).

```julia
# Default NUTS settings
result = fit(problem, MCMCNUTS())

# Custom settings
result = fit(
    problem, 
    MCMCNUTS(; target_acceptance=0.9, max_depth=12);
    n_samples=2000,
    n_chains=4
)
```

**When to use:**
- Small to medium datasets (n < 1000)
- When exact inference is critical
- For final published results

**Key parameters:**
- `target_acceptance`: Target acceptance rate (default: 0.8, range: 0.6-0.9)
- `max_depth`: Maximum tree depth (default: 10)
- `n_samples`: Number of posterior samples per chain (default: 2000)
- `n_chains`: Number of parallel chains (default: 4)

### Simple VI with Mooncake

Simple VI uses Turing's native ADVI implementation and works excellently with the Mooncake AD backend for 5-10x speedup after warmup.

```julia
# Default: SimpleVI with Mooncake
result = fit(problem, SimpleVI())

# Or explicitly
result = fit(
    problem, 
    SimpleVIMethod(; ad_backend=AutoMooncake);
    n_iterations=1000,
    n_draws=2000
)
```

**When to use:**
- Production environments
- When you can afford a warmup run
- Small to medium data where VI approximation is acceptable

**Performance tip:** Mooncake requires compilation on first use. Run a short warmup:

```julia
# Warmup run (slow - compiles differentiation rules)
result_warmup = fit(
    problem, 
    SimpleVIMethod(; ad_backend=AutoMooncake);
    n_iterations=50
)

# Production runs (fast - 5-10x faster than ReverseDiff)
result = fit(
    problem, 
    SimpleVIMethod(; ad_backend=AutoMooncake);
    n_iterations=1000
)
```

### Unified VI with AutoReverseDiff

Unified VI uses AdvancedVI.jl with explicit bijectors and supports multiple variational families and subsampling for large datasets.

#### MeanField (Diagonal Covariance)

The default MeanField approximation assumes independent parameters:

```julia
# MeanField with ReverseDiff (default)
result = fit(
    problem,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=MeanField()
    );
    n_iterations=1000,
    n_draws=2000
)

# Or use the convenience constructor
result = fit(problem, MeanFieldVI())
```

#### LowRank (Low-Rank + Diagonal Covariance)

LowRank captures parameter correlations with fewer parameters than full covariance:

```julia
# LowRank with rank 3
result = fit(
    problem,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=LowRank(3)
    );
    n_iterations=1000,
    n_draws=2000
)

# Or use the convenience constructor
result = fit(problem, LowRankVI(3))
```

**When to use:**
- Large datasets (automatically enables subsampling when n ≥ 10,000)
- When you need specific variational family control
- For exploring mean-field vs low-rank tradeoffs

**Subsampling:**
Automatically enabled for n ≥ 10,000:

```julia
# Auto-subsampling (default batch size: min(256, ceil(n/1000)))
result = fit(problem, UnifiedVIMethod())  # Auto-enabled for large n

# Explicit control
result = fit(
    problem,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=MeanField(),
        subsample=true,
        batch_size=512
    );
    n_iterations=1000
)
```

## Understanding Results

All inference methods return result objects that support common accessor functions:

```julia
# Extract causal effect samples
alpha_samples = result.alpha_samples  # On original scale
alpha_std = result.alpha_samples_standardized  # On standardized scale

# Generate coefficient table with diagnostics
coeftable(result)
```

### Coefficient Table Output

```
Bayesian Double ML Coefficient Table
======================================================================
Parameter: α (treatment effect)
Model type: hier
Inference method: VI
Credible interval level: 95.0% (HPD)
Number of posterior samples: 2000

  Parameter    Estimate  Std. Error        MCSE     P-value
  ---------    --------  ----------        ----     -------
          α        1.9832      0.1124      0.0025      0.0000

HPD Credible Intervals:
  α: [1.7623, 2.2031]

Diagnostics:
  Final ELBO: -3421.56
```

### MCMC-Specific Diagnostics

```julia
# Effective Sample Size (ESS)
ess(result)

# R-hat convergence diagnostic (should be ≈ 1.0)
rhat(result)

# Monte Carlo Standard Error
mcse(result)
```

### VI-Specific Information

```julia
# ELBO convergence history
result.elbo_history

# Convergence flag
result.converged

# Final ELBO value
result.final_elbo
```

## Performance Tips

### AD Backend Selection

| Backend | Speed | Stability | Warmup | Best For |
|---------|-------|-----------|--------|----------|
| AutoReverseDiff | Baseline | Excellent | None | Default choice |
| AutoMooncake | 5-10x faster | Good | Required | Production/batch |
| AutoZygote | Variable | Good | None | Experimentation |
| AutoForwardDiff | Slow for large p | Excellent | None | Small models (p < 20) |

### Dataset Size Guidelines

| Size (n) | Recommended Method | Notes |
|----------|-------------------|-------|
| n < 1000 | MCMCNUTS() | Exact inference |
| 1000 ≤ n < 10000 | SimpleVI() with Mooncake | Fast with warmup |
| n ≥ 10000 | UnifiedVI() with subsampling | Memory efficient |

### Memory Considerations

For very large datasets:
- MCMC chains consume significant memory
- Use VI methods for memory efficiency
- Enable subsampling in UnifiedVI for n > 10,000

## Mathematical Background

### The Bivariate Reduced Form

The BDML model avoids regularization-induced confounding by parameterizing the causal inference problem as a bivariate regression:

**Structural Model:**
```math
Y = \alpha D + X'\beta + \varepsilon, \quad \varepsilon \perp V
```

**Reduced Form (substituting D = X'\gamma + V):**
```math
\begin{aligned}
Y &= X'\underbrace{(\beta + \alpha\gamma)}_{\delta} + \underbrace{(\varepsilon + \alpha V)}_{U} \\
D &= X'\gamma + V
\end{aligned}
```

Since ``\varepsilon \perp V`` by assumption:
```math
\text{Cov}(U, V) = \text{Cov}(\varepsilon + \alpha V, V) = \alpha \cdot \text{Var}(V)
```

Therefore, the causal effect is:
```math
\alpha = \frac{\text{Cov}(U, V)}{\text{Var}(V)} = \frac{\sigma_{UV}}{\sigma^2_V} = \rho \frac{\sigma_U}{\sigma_V}
```

### Prior Specifications

**BDML-Basic:**
```math
\begin{aligned}
\delta &\sim \mathcal{N}(0, 25 \cdot I_p) \\
\gamma &\sim \mathcal{N}(0, 25 \cdot I_p) \\
\sigma_U, \sigma_V &\sim \text{Cauchy}^+(0, 2.5) \\
R &\sim \text{LKJ}(4)
\end{aligned}
```

**BDML-Hier:**
```math
\begin{aligned}
\sigma^2_\delta, \sigma^2_\gamma &\sim \text{InvGamma}(2, 2) \\
\delta \mid \sigma^2_\delta &\sim \mathcal{N}(0, \sigma^2_\delta \cdot I_p) \\
\gamma \mid \sigma^2_\gamma &\sim \mathcal{N}(0, \sigma^2_\gamma \cdot I_p) \\
\sigma_U, \sigma_V &\sim \text{Cauchy}^+(0, 2.5) \\
R &\sim \text{LKJ}(4)
\end{aligned}
```

The hierarchical prior is equivalent to placing independent Student-t(4) distributions on each coefficient marginally, providing adaptive shrinkage that learns the appropriate regularization from data.

## References

- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for Causal Inference". arXiv:2508.12688v1.
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters". The Econometrics Journal, 21(1), C1-C68.
