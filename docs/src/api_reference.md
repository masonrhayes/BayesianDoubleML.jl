# API Reference

Complete API documentation for BayesianDoubleML.jl.

## Contents

- [Model Types](#model-types)
- [Inference Methods](#inference-methods)
- [Fitting Functions](#fitting-functions)
- [Result Types](#result-types)
- [Extraction Functions](#extraction-functions)
- [Statistical Functions](#statistical-functions)
- [Coefficient Table](#coefficient-table)
- [Utility Functions](#utility-functions)

## Model Types

```@docs
BayesianDoubleML.AbstractBDMLModel
BayesianDoubleML.BDMLBasicModel
BayesianDoubleML.BDMLHierarchicalModel
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
BayesianDoubleML.UnifiedVIMethod
BayesianDoubleML.SimpleVIMethod
```

### MCMC Method Constructors

```@docs
BayesianDoubleML.MCMCNUTS
```

### VI Method Constructors

```@docs
BayesianDoubleML.UnifiedVI
BayesianDoubleML.SimpleVI
BayesianDoubleML.MeanFieldVI
BayesianDoubleML.LowRankVI
BayesianDoubleML.LowRankScoreVI
```

### Variational Families

```@docs
BayesianDoubleML.AbstractVariationalFamily
BayesianDoubleML.MeanField
BayesianDoubleML.LowRank
BayesianDoubleML.LowRankScore
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

## Utility Functions

```@docs
BayesianDoubleML.credible_interval
BayesianDoubleML.check_convergence
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
BayesianDoubleML.generate_dgp_table1
```

## Index

```@index
```
