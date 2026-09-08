# API Reference


## Model Types

```@docs
BayesianDoubleML.AbstractBDMLModel
BayesianDoubleML.BDMLBasicModel
BayesianDoubleML.BDMLHierarchicalModel
BayesianDoubleML.BayesDRModel
BayesianDoubleML.BDMLModel
```

### Model Accessors

```@docs
BayesianDoubleML.nobs
BayesianDoubleML.ncovariates
BayesianDoubleML.model_type
BayesianDoubleML.standardization_stats
BayesianDoubleML.isfitted
```

## Inference Methods

### Abstract and Types

```@docs
BayesianDoubleML.AbstractInferenceMethod
BayesianDoubleML.MCMCMethod
BayesianDoubleML.CollapsedVIMethod
BayesianDoubleML.VMPMethod
BayesianDoubleML.BayesDRMCMCMethod
BayesianDoubleML.AbstractVMPBackend
BayesianDoubleML.RxInferVMP
BayesianDoubleML.ManualCoordinateAscentVMP
```

### MCMC Method Constructors

```@docs
BayesianDoubleML.MCMCNUTS
```

### VI/VMP Method Constructors

```@docs
BayesianDoubleML.CollapsedVI
BayesianDoubleML.VMP
BayesianDoubleML.BayesDRMCMC
```

### Method Traits

```@docs
BayesianDoubleML.uses_sampling
BayesianDoubleML.supports_subsampling
BayesianDoubleML.is_deterministic
BayesianDoubleML.default_n_samples
BayesianDoubleML.default_n_iterations
```

## Fitting Functions

```@docs
BayesianDoubleML.fit!
```

## Result Types

```@docs
BayesianDoubleML.AbstractBDMLResult
BayesianDoubleML.BDMLMCMCResult
BayesianDoubleML.BDMLVIResult
BayesianDoubleML.BDMLVMPResult
BayesianDoubleML.BayesDRResult
BayesianDoubleML.BayesDRCurveResult
BayesianDoubleML.BDMLData
BayesianDoubleML.StandardizationStats
```

## Extraction Functions

```@docs
BayesianDoubleML.extract_alpha
```

## Statistical Functions

### Coefficient Table

```@docs
BayesianDoubleML.coeftable
BayesianDoubleML.BDMLCoeftable
```

### Diagnostics

```@docs
BayesianDoubleML.confint
BayesianDoubleML.ess
BayesianDoubleML.pvalues
BayesianDoubleML.hpd_interval
BayesianDoubleML.mcse
BayesianDoubleML.rhat
BayesianDoubleML.rhat_statistic
BayesianDoubleML.chain_info
```

### StatsAPI Functions

```@docs
BayesianDoubleML.coef
BayesianDoubleML.stderror
BayesianDoubleML.vcov
```

### BayesDR Functions

```@docs
BayesianDoubleML.exposure_response_curve
BayesianDoubleML.average_derivative
```

## Utility Functions

```@docs
BayesianDoubleML.credible_interval
```

## Visualization

Requires a Makie backend (e.g. `CairoMakie`). Loaded via package extension.

```@docs
BayesianDoubleML.plot_exposure_response_curve
```

## Internal Functions

These functions are primarily for internal use but are documented here for developers.

### Model Functions

```@docs
BayesianDoubleML.bdml_basic
BayesianDoubleML.bdml_hier
```

### Data Functions

```@docs
BayesianDoubleML.make_plr_DTL2025
BayesianDoubleML.make_irm_APD2022
BayesianDoubleML.make_er_APD2022
BayesianDoubleML.make_plr_LML2025
```

## Index

```@index

```
