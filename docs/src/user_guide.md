# User Guide

This guide demonstrates how to use BayesianDoubleML.jl for causal inference with different inference methods.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/BayesianDoubleML.jl")
```

## Basic Usage

### Creating a Model

All analyses start by creating a `BDMLModel` from your data:

```julia
using BayesianDoubleML
using StableRNGs

# Your data
n = 200
p = 100
alpha_true = 2.0 # true causal effect
rng = StableRNG(42) # for reproducibility

# Generate data as DataFrame
df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)

# Create model with hierarchical model (recommended)
# All columns except :y and :d are automatically used as covariates
model = BDMLModel(df, :y, :d; model_type=:hier)

# Or use basic model with fixed priors
model = BDMLModel(df, :y, :d; model_type=:basic)

# Explicit covariate selection (optional)
# model = BDMLModel(df, :y, :d; model_type=:hier, x_cols=[:X1, :X2, :X3])
```

### Model Types

**Hierarchical Model (`:hier`) - Recommended:**

- Uses adaptive shrinkage via hierarchical priors
- Equivalent to Student-t(4) priors on coefficients
- Better coverage in simulations (0.94 vs 0.91-0.93) from the paper.

**Basic Model (`:basic`):**

- Uses fixed N(0, 25·I) priors
- Good for baseline comparisons

## Inference Methods

### MCMC (NUTS)

MCMC using the No-U-Turn Sampler provides great posterior inference and is recommended for small-to-medium datasets or where fitting time is not a large concern.

```julia
# Default NUTS settings
fit!(model, MCMCNUTS())

# Custom settings
fit!(
    model, 
    MCMCNUTS(; target_acceptance=0.9, max_depth=12);
    n_samples=2000,
    n_chains=4
)
```

**When to use:**

- Small to medium datasets
- When exact inference is critical

**Key parameters:**

- `target_acceptance`: Target acceptance rate (default: 0.8, range: 0.6-0.9)
- `max_depth`: Maximum tree depth (default: 10)
- `n_samples`: Number of posterior samples per chain (default: 2000)
- `n_chains`: Number of parallel chains (default: 4)

### Simple VI with Mooncake

Simple VI uses Turing's native ADVI implementation and works excellently with the Mooncake AD backend, which provides  5-10x speedup (after initial warmup).

```julia
# Default: SimpleVI with Mooncake
using Mooncake 

fit!(
    model, 
    SimpleVIMethod(; ad_backend=AutoMooncake);
    n_iterations=1000,
    n_draws=2000
)
```

### Unified VI with AutoReverseDiff

Unified VI uses AdvancedVI.jl with explicit bijectors and supports multiple variational families and subsampling for large datasets.

#### MeanField (Diagonal Covariance)

The default MeanField approximation assumes independent parameters:

```julia
# MeanField with ReverseDiff (default)
fit!(
    model,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=MeanField()
    );
    n_iterations=1000,
    n_draws=2000
)

# Or use the convenience constructor
fit!(model, MeanFieldVI())
```

#### LowRank (Low-Rank + Diagonal Covariance)

LowRank captures parameter correlations with fewer parameters than full covariance:

```julia
# LowRank with rank 3
fit!(
    model,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=LowRank(3)
    );
    n_iterations=1000,
    n_draws=2000
)

# Or use the convenience constructor
fit!(model, LowRankVI(3))
```

**When to use:**

- Large datasets (automatically enables subsampling when n ≥ 10,000)
- When you need specific variational family control
- For exploring mean-field vs low-rank tradeoffs

**Subsampling:**
Automatically enabled for n ≥ 10,000:

```julia
# Auto-subsampling (default batch size: min(256, ceil(n/1000)))
fit!(model, UnifiedVIMethod())  # Auto-enabled for large n

# Explicit control
fit!(
    model,
    UnifiedVIMethod(; 
        ad_backend=AutoReverseDiff,
        family=MeanField(),
        subsample=true,
        batch_size=512
    );
    n_iterations=1000
)
```

### Variational Message Passing (VMP)

VMP uses a conjugate reparameterization of the BDML reduced form so that all
message updates are closed-form. The default backend requires `RxInfer.jl`;
a manual coordinate-ascent backend is also available and works without any
optional dependency.

```julia
# Default RxInfer backend (requires `using RxInfer`)
fit!(model, VMP(); n_iterations = 50)

# Manual backend (no optional dependencies)
fit!(model, VMP(; backend = ManualCoordinateAscentVMP()); n_iterations = 50)

# Custom prior hyperparameters
method = VMP(;
    backend = RxInferVMP(),
    ν0 = 4.0,
    S0 = [1.0 0.0; 0.0 1.0],
)
fit!(model, method; n_iterations = 50)
```

## Understanding Results

All inference methods store results in the model object. Access them using common accessor functions:

```julia
# Extract causal effect samples
alpha_samples = extract_alpha(model)  # On original scale

# Generate coefficient table with diagnostics
coeftable(model)

# See full model result summary
summary(model)
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
ess(model)

# R-hat convergence diagnostic (should be ≈ 1.0)
rhat(model)

# Monte Carlo Standard Error
mcse(model)
```

### VI-Specific Information

```julia
# Access the result object for VI-specific fields
result = model.result

# ELBO convergence history
result.elbo_history

# Convergence flag
result.converged

# Final ELBO value
result.final_elbo
```

## Performance Tips

### AD Backend Selection

| Backend         | Speed                                         |  Best For        |
| --------------- | --------------------------------------------- |  --------------- |
| AutoReverseDiff | Baseline                                      |  Default choice  |
| AutoMooncake    | 5-10x faster                                  |  Speed           |
| AutoZygote      | Typically slower than ReverseDiff or Mooncake |  Not recommended |
| AutoForwardDiff | Typically very slow                           |  Not recommended |

## Mathematical Background

### The Bivariate Reduced Form

The BDML model avoids regularization-induced confounding by parameterizing the causal inference problem as a bivariate regression:

**Structural Model:**

```math
Y = \alpha D + X'\beta + \varepsilon, \quad \varepsilon \perp V
```

**Reduced Form (substituting $D = X'\gamma + V$):**

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
