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
df = make_plr_DTL2025(
    rng; n, p, sigma_epsilon = 2.0, alpha = alpha_true,
)

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

### Collapsed VI

Collapsed VI integrates the high-dimensional coefficients out analytically
and runs ADVI on the 3-dimensional (`:basic`) or 5-dimensional (`:hier`)
marginal posterior for the error scales, correlation, and optional
coefficient variances.

```julia
# Default: CollapsedVI with ReverseDiff (full-rank Gaussian)
fit!(
    model,
    CollapsedVI();
    n_iterations=1000,
    n_draws=2000
)

# Mean-field family with the Mooncake AD backend (faster after warmup)
using Mooncake

fit!(
    model,
    CollapsedVI(; ad_backend=AutoMooncake, fullrank=false);
    n_iterations=1000,
    n_draws=2000
)
```

**When to use:**

- Medium to large datasets where MCMC is too slow
- When you want a fast posterior approximation with ELBO diagnostics

**Key parameters:**

- `ad_backend`: AD backend (default: `AutoReverseDiff`)
- `n_montecarlo`: Monte Carlo samples per ELBO gradient (default: 10)
- `fullrank`: `true` for a full-rank Gaussian, `false` for mean-field (default: `true`)

### Variational Message Passing (VMP)

VMP uses a conjugate reparameterization of the BDML reduced form so that all
message updates are closed-form. The default manual coordinate-ascent backend
works without optional dependencies. An RxInfer backend is also available when
`RxInfer.jl` is loaded.

VMP replaces the LKJ(4) + Half-Cauchy prior on the error covariance with the conjugate `InverseWishart(ν0, S0)` prior from the paper's theoretical
specification (Equation 19). The default `S0 = nothing` uses a data-driven
diagonal scaling on the standardized scale; pass an explicit 2×2 symmetric
positive-definite `S0` to override it. The `aτ`/`bτ` Gamma hyperprior only
affects the `:hier` model.

Reported `α` uncertainty applies an effective residual degrees-of-freedom
correction to the mean-field covariance posterior. In `result.posterior`, `Σ`
is calibrated for `α` (not an exact joint covariance posterior), `Σ_vmp`
retains the raw variational posterior, and `effective_df` records the
correction. `summary` also reports `Effective DF`.

```julia
# Default manual backend (no optional dependencies)
fit!(model, VMP(); n_iterations = 50)

# RxInfer backend (requires `using RxInfer`)
using RxInfer
fit!(model, VMP(; backend = RxInferVMP()); n_iterations = 50)

# Custom prior hyperparameters
method = VMP(;
    backend = RxInferVMP(),
    ν0 = 4.0,
    S0 = [1.0 0.0; 0.0 1.0],
)
fit!(model, method; n_iterations = 50, n_draws = 2000)
```

**When to use:**

- Large `p` where MCMC/CollapsedVI are too slow and `p < n`
- Fast approximate inference with ELBO/Bethe free-energy diagnostics

**Key parameters:**

- `backend`: `ManualCoordinateAscentVMP()` (default) or `RxInferVMP()`
- `ν0`: Inverse-Wishart prior degrees of freedom (default: `4.0`, must exceed 3)
- `S0`: Inverse-Wishart scale matrix (default: `nothing` for data-driven scaling)
- `n_iterations`: coordinate-ascent sweeps (default: `50`)
- `n_draws`: posterior `α` draws after fitting (default: `2000`)

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

Below is a CollapsedVI example (`method_type` prints as `VI` with `MCSE 0.0000`
because draws are independent). VMP prints `Inference method: VMP` with
`Final Diagnostic` instead of `Final ELBO`; see `summary` for its
`Effective DF`.

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
          α        1.9832      0.1124      0.0000      0.0000

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
# Access the result object for CollapsedVI-specific fields
result = model.result

# ELBO convergence history
result.elbo_history

# Convergence flag
result.converged

# Final ELBO value
result.final_elbo
```

### VMP-Specific Information

```julia
# Access the result object for VMP-specific fields
result = model.result

# ELBO / negative Bethe free-energy trace (manual / RxInfer)
result.diagnostic_history
result.final_diagnostic
result.diagnostic_kind

# Effective residual-DF correction for α uncertainty
result.posterior.effective_df
result.posterior.Σ      # calibrated for α
result.posterior.Σ_vmp  # raw variational covariance posterior
```

## Performance Tips

### AD Backend Selection

| Backend         | Speed                                         | Best For        |
| --------------- | --------------------------------------------- | --------------- |
| AutoReverseDiff | Baseline                                      | Default choice  |
| AutoMooncake    | 5-10x faster                                  | Speed           |
| AutoZygote      | Typically slower than ReverseDiff or Mooncake | Not recommended |
| AutoForwardDiff | Typically very slow                           | Not recommended |

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

## Experimental Bayes-DR

`BayesDRModel` implements the binary-treatment ATE and continuous-treatment
exposure-response procedures of Antonelli, Papadogeorgou, and Dominici (2022).
It fits separate spike-and-slab Bayesian nuisance models and posterior-averages
a doubly robust estimator.

```julia
model = BayesDRModel(Y, T, X)
fit!(model, BayesDRMCMC(); n_samples = 1000, n_burn = 500, n_boot = 500)
coeftable(model)
```

Binary `T` must contain both `0` and `1`; a treatment with more than two levels
is handled as continuous. Covariates are standardized internally, while curve
locations and the continuous outcome remain on their original scales.
The reported interval is a frequentist confidence interval whose variance is
the sum of a bootstrap data component and a posterior nuisance-parameter
component. Consequently, `credible_interval` and `extract_alpha` are not
defined for this model.

The current experimental implementation uses additive nuisance models. Spline,
Gaussian-process, and binary-outcome variants from the paper remain future
extensions.

The nuisance models use a custom conjugate Gibbs sampler. This retains the
exact point-mass spike-and-slab updates and supports high-dimensional settings
without requiring a mixed discrete/continuous Turing sampler. Common MCMC
diagnostics summarize both nuisance fits conservatively:

```julia
ess(model)        # minimum nuisance-parameter ESS
rhat(model)       # maximum nuisance-parameter R-hat, or missing for one chain
mcse(model)       # maximum nuisance-parameter MCSE
chain_info(model)
```

The complete treatment and outcome inclusion draws are available through
`model.result.treatment_posterior.inclusion` and
`model.result.outcome_posterior.inclusion`. Average them over the first
dimension to obtain marginal inclusion probabilities.

Small binary and continuous characterization comparisons against the reference
R package live in `test/extended/bayes_dr_reference.jl`. They cover estimates,
standard errors, and intervals with broad Monte Carlo tolerances. The committed
fixtures can be regenerated manually with
`Rscript test/reference/generate_bayes_dr_reference.R`; R is not invoked by the
normal test suite.

**Reference**: Antonelli, Papadogeorgou, and Dominici (2022), [doi:10.1111/biom.13417](https://doi.org/10.1111/biom.13417).

The paper's binary-treatment simulation design is available in both its linear and nonlinear forms:

```julia
df_linear = make_irm_APD2022(
    StableRNG(42); n = 100, p = 500, scenario = :linear,
)
df_nonlinear = make_irm_APD2022(
    StableRNG(42); n = 100, p = 500, scenario = :nonlinear,
)
```

Both designs have true ATE 1 by default. Pass `alpha` to change it and `rng` to
make simulation replications reproducible.

The paper's continuous-treatment exposure-response design is available with
its defaults of `n = 200` and `p = 200`:

```julia
df_curve = make_er_APD2022(StableRNG(42))
model = BayesDRModel(df_curve, :y, :d)
```

Pass signed `cubic_coefficient` and `quadratic_coefficient` values to customize
the nonlinear treatment response. Their paper defaults are `0.05` and `-0.1`.

The continuous-exposure simulation from Luo et al. (2025), Section 3.2, is
available with its paper settings of `n = 40`, `p = 40`, and true effect 1:

```julia
df_continuous = make_plr_LML2025(StableRNG(42))
model = BDMLModel(df_continuous, :y, :d)
```

Pass `alpha` to customize the constant treatment effect in either LML2025
treatment design.

Pass `treatment = :binary` for the Section 3.1 design. Its paper settings use
`p = 500` and `n = 50` or `n = 200`:

```julia
df_binary = make_plr_LML2025(
    StableRNG(42); n = 200, p = 500, treatment = :binary,
)
```

## References

- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for Causal Inference". arXiv:2508.12688v1.
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters". The Econometrics Journal, 21(1), C1-C68.
- Luo, M., Moodie, E. E. M., Bhatnagar, S., & Lee, D. (2025). "A scalable Bayesian double machine learning framework, with application to racial disproportionality".
