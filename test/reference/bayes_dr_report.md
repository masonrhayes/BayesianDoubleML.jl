# Bayes-DR Julia–R Reference Report

Comparison of Julia `BayesDRModel` fits against `DoublyRobustHD` version `0.0.0.9000`, Git commit `617945098b2f540039367d4398068ba9ae713891`.

- Julia package: BayesianDoubleML v0.2.0
- Sampler: `BayesDRMCMC` with 250 retained draws per chain, 500 burn-in iterations, thinning 2, 2 chains, 500 bootstrap replicates
- Fixtures: `bayes_dr_binary.csv` (n = 30, binary treatment) and `bayes_dr_continuous.csv` (n = 24, continuous treatment)
- Tolerances are broad Monte Carlo bounds; these are characterization checks, not exact equivalence tests

## Binary treatment

| Quantity | R reference | Julia | Absolute difference | Tolerance | Status |
|---|---:|---:|---:|---:|---|
| Estimate | 0.5810 | 0.5972 | 0.0162 | 0.35 | pass |
| Standard error | 0.3044 | 0.2557 | 0.0487 | 0.35 | pass |
| CI lower bound | 0.1393 | 0.1637 | 0.0243 | 0.6 | pass |
| CI upper bound | 1.2882 | 1.1738 | 0.1144 | 0.6 | pass |

## Continuous treatment

| Quantity | Location | R reference | Julia | Absolute difference | Tolerance | Status |
|---|---:|---:|---:|---:|---:|---|
| Estimate | -0.5000 | 1.0962 | 1.1545 | 0.0583 | 0.6 | pass |
| Estimate | 0.0000 | 1.3532 | 1.4155 | 0.0623 | 0.6 | pass |
| Estimate | 0.5000 | 1.5223 | 1.5430 | 0.0207 | 0.6 | pass |
| Standard error | -0.5000 | 0.9623 | 0.9620 | 0.0003 | 0.6 | pass |
| Standard error | 0.0000 | 0.8605 | 0.8273 | 0.0332 | 0.6 | pass |
| Standard error | 0.5000 | 0.4542 | 0.2952 | 0.1590 | 0.6 | pass |
| CI lower bound | -0.5000 | 0.1691 | 0.1326 | 0.0366 | 0.9 | pass |
| CI lower bound | 0.0000 | 0.5973 | 0.5612 | 0.0361 | 0.9 | pass |
| CI lower bound | 0.5000 | 1.0289 | 1.1047 | 0.0758 | 0.9 | pass |
| CI upper bound | -0.5000 | 2.7925 | 2.8914 | 0.0989 | 0.9 | pass |
| CI upper bound | 0.0000 | 2.7930 | 2.8767 | 0.0837 | 0.9 | pass |
| CI upper bound | 0.5000 | 2.2937 | 2.1874 | 0.1063 | 0.9 | pass |
