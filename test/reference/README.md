# Bayes-DR R Reference Fixtures

These small fixtures compare Julia point estimates with `DoublyRobustHD`
version `0.0.0.9000`, Git commit
`617945098b2f540039367d4398068ba9ae713891`.

Regenerate them from the repository root with:

```bash
Rscript test/reference/generate_bayes_dr_reference.R
```

The comparisons cover point estimates, standard errors, and confidence
intervals with broad Monte Carlo tolerances. They are characterization checks,
not exact equivalence tests; the expected uncertainty values may be updated
when the Bayes-DR uncertainty implementation is reviewed separately.

Running `test/extended/bayes_dr_reference.jl` writes a Markdown comparison
report to `bayes_dr_report.md` in this directory, listing the R reference
values, the Julia estimates, absolute differences, tolerances, and pass/fail
status for every quantity. The report is regenerated on each run; commit an
updated copy alongside the fixtures when they change.

Both fixtures retain 250 posterior draws per chain after 500 burn-in
iterations and use 500 bootstrap replicates. Dataset dimensions are kept small
to limit the runtime of the reference R implementation.
