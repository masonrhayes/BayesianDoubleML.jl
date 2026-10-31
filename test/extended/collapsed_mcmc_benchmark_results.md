# Full vs. collapsed BDML NUTS: experimental run

**Run:** 2026-09-27, Julia 1.13.0, BayesianDoubleML.jl 0.2.1 (working tree with the collapsed-MCMC implementation), Raptor Lake CPU, 2 Julia threads, **1 BLAS thread**.

```bash
julia -t 2 --project=test test/extended/collapsed_mcmc_benchmark.jl
```

This is the standalone experiment in [`collapsed_mcmc_benchmark.jl`](collapsed_mcmc_benchmark.jl), **not part of either test suite**. The data were generated **once** using `make_plr_DTL2025(200, 100, 2.0; alpha=2.0, rng=StableRNG(42))`: one replication of the DiTraglia–Liu simulation setup, with true α = 2 (DiTraglia & Liu, 2025, §6). Both model variants (`:basic` and `:hier`) and both inference methods use that same dataset. Both use NUTS with target acceptance `0.8`, maximum depth `10`, 2 chains × 500 retained draws, and a reset sampler seed of `2026` before each fit. Their parameter dimensions differ, so equal seeds **do not** imply identical trajectories.

## Measured results

| Model | Method | Fit time (s) | α mean | α SD | 95% equal-tailed α interval | α ESS | α R̂ | α ESS/s |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Basic | Full NUTS | 38.32 | 1.969 | 0.201 | [1.572, 2.353] | 767.4 | 0.998 | 20.0 |
| Basic | Collapsed NUTS | ~0.02 | 1.966 | 0.210 | [1.539, 2.386] | 689.5 | 1.004 | 38,766.8 |
| Hierarchical | Full NUTS | 29.27 | 1.912 | 0.191 | [1.533, 2.285] | 760.4 | 1.005 | 26.0 |
| Hierarchical | Collapsed NUTS | ~0.02 | 1.914 | 0.195 | [1.519, 2.278] | 956.9 | 1.000 | 41,993.5 |

| Model | Fit-time ratio (full / collapsed) | α ESS/s ratio (collapsed / full) | Experiment checks |
| --- | ---: | ---: | --- |
| Basic | 2,154.74× | 1,936.02× | 7/7 passed |
| Hierarchical | 1,284.37× | 1,616.20× | 7/7 passed |

Both posterior means differed by less than 0.004 on this dataset; the posterior SDs and interval endpoints were also close. Both methods' intervals contained the generating α = 2. The experiment checks finite diagnostics, posterior mean/SD/interval agreement, faster fit time, and higher α ESS per second for each variant.

## What the timings do and do not mean

- The script warms up **both** samplers before timing. Timed `fit!` includes collapsed SVD construction, NUTS adaptation, and posterior sampling; it excludes `BDMLModel` construction/standardization, initial compilation, and the separate computation of diagnostics.
- Full NUTS time is **one** timed fit per model variant. Collapsed NUTS time is the **median of 10** timed fits with the same settings, because each takes only milliseconds. The printed `0.02 s` values are rounded; the ratios use unrounded times. Timings this small are sensitive to machine load and clock resolution.
- ESS and R̂ above are computed from the *derived α draws*, reshaped by chain, rather than taking the most favorable nuisance parameter. ESS/s divides those values by the respective measured fit time.
- This is one dataset and one run on one computer, **not a paper-level runtime claim**. The ratios compare these **Julia/Turing implementations**, not the paper's Stan implementation. The agreement thresholds are practical Monte Carlo checks, not proof of equality from finite chains; equality of the ideal α posterior follows separately from Gaussian marginalization under matching priors.
- Replicate with multiple data seeds and longer chains, monitor divergences, and measure compilation, peak memory, and the effect of rank/$p$ before presenting any general performance figure to the authors.

## Reference

DiTraglia, F. J., & Liu, L. (2025). *Bayesian double machine learning for causal inference* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.2508.12688
