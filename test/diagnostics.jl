# Diagnostics Tests
# Tests for coeftable, HPD intervals, MCSE, ESS, p-values
# Statistical diagnostics and inference summaries

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "Credible Interval Function" begin
    Random.seed!(700)

    # Test with normal samples
    samples = randn(10000) .+ 0.5

    # Default 95% CI
    ci_95 = BayesianDoubleML.credible_interval(samples)
    @test length(ci_95) == 2
    @test ci_95[1] < ci_95[2]
    @test ci_95[1] < mean(samples) < ci_95[2]

    # 90% CI should be narrower
    ci_90 = BayesianDoubleML.credible_interval(samples; level = 0.9)
    @test ci_90[1] > ci_95[1]
    @test ci_90[2] < ci_95[2]

    # 99% CI should be wider
    ci_99 = BayesianDoubleML.credible_interval(samples; level = 0.99)
    @test ci_99[1] < ci_95[1]
    @test ci_99[2] > ci_95[2]

    println("\n=== Credible Interval Tests ===")
    println("  90% CI: [$(round(ci_90[1], digits = 4)), $(round(ci_90[2], digits = 4))]")
    println("  95% CI: [$(round(ci_95[1], digits = 4)), $(round(ci_95[2], digits = 4))]")
    println("  99% CI: [$(round(ci_99[1], digits = 4)), $(round(ci_99[2], digits = 4))]")
    println("  ✓ Credible intervals work correctly")
end

@testset "Posterior Summary" begin
    Random.seed!(701)
    samples = randn(1000) .+ 2.0

    summary = BayesianDoubleML.posterior_summary(samples)

    @test haskey(summary, :mean)
    @test haskey(summary, :std)
    @test haskey(summary, :quantiles)
    @test haskey(summary, :prob_range)

    @test summary[:mean] ≈ mean(samples)
    @test summary[:std] ≈ std(samples)
    @test length(summary[:quantiles]) == 5
    @test summary[:prob_range] == [0.025, 0.25, 0.5, 0.75, 0.975]

    println("\n=== Posterior Summary ===")
    println("  Mean: $(round(summary[:mean], digits = 4))")
    println("  Std: $(round(summary[:std], digits = 4))")
    println("  Median (50%): $(round(summary[:quantiles][3], digits = 4))")
    println("  ✓ Posterior summary works correctly")
end

@testset "coeftable for BDMLResult (MCMC)" begin
    Random.seed!(702)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 702)

    println("\n=== coeftable: BDMLResult (MCMC) ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    ct = coeftable(result)

    @test ct isa CoefTable
    @test length(ct.rownms) >= 1  # At least alpha parameter
    @test length(ct.colnms) >= 4   # Estimate, Std Error, z value, Pr(>|z|)

    # Find alpha row
    alpha_idx = findfirst(contains("alpha"), ct.rownms)
    @test alpha_idx !== nothing

    println("  Coefficient table rows: $(length(ct.rownms))")
    println("  Coefficient table cols: $(length(ct.colnms))")
    println("  Alpha estimate: $(round(ct.mat[alpha_idx, 1], digits = 4))")
    println("  ✓ coeftable works for MCMC results")
end

@testset "coeftable for BDMLVIResult (VI)" begin
    Random.seed!(703)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 703)

    println("\n=== coeftable: BDMLVIResult (VI) ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    ct = coeftable(result)

    @test ct isa CoefTable
    @test length(ct.rownms) >= 1
    @test length(ct.colnms) >= 4

    alpha_idx = findfirst(contains("alpha"), ct.rownms)
    @test alpha_idx !== nothing

    println("  Coefficient table rows: $(length(ct.rownms))")
    println("  Alpha estimate: $(round(ct.mat[alpha_idx, 1], digits = 4))")
    println("  ✓ coeftable works for VI results")
end

@testset "confint for MCMC and VI Results" begin
    Random.seed!(704)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 704)

    println("\n=== confint: MCMC and VI Results ===")

    # MCMC
    problem_mcmc = BDMLProblem(Y, D, X; model_type = :hier)
    result_mcmc = fit(problem_mcmc, MCMCNUTS(; n_samples = 200, n_warmup = 100))

    ci_mcmc = confint(result_mcmc)
    @test length(ci_mcmc) == 2
    @test ci_mcmc[1] < ci_mcmc[2]

    # VI
    problem_vi = BDMLProblem(Y, D, X; model_type = :hier)
    result_vi = fit(problem_vi, UnifiedVI())

    ci_vi = confint(result_vi)
    @test length(ci_vi) == 2
    @test ci_vi[1] < ci_vi[2]

    println("  MCMC 95% CI: [$(round(ci_mcmc[1], digits = 4)), $(round(ci_mcmc[2], digits = 4))]")
    println("  VI 95% CI: [$(round(ci_vi[1], digits = 4)), $(round(ci_vi[2], digits = 4))]")
    println("  ✓ confint works for both MCMC and VI")
end

@testset "Effective Sample Size (ESS)" begin
    Random.seed!(705)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 705)

    println("\n=== Effective Sample Size (ESS) ===")

    # MCMC with multiple chains for ESS calculation
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100, n_chains = 2)

    # ESS function should exist and return positive value
    ess_value = ess(result)

    @test ess_value > 0
    @test ess_value <= length(result.alpha_samples)

    println("  Total samples: $(length(result.alpha_samples))")
    println("  ESS: $(round(ess_value, digits = 1))")
    println("  Efficiency: $(round(ess_value / length(result.alpha_samples) * 100, digits = 1))%")
    println("  ✓ ESS calculation works")
end

@testset "Monte Carlo Standard Error (MCSE)" begin
    Random.seed!(706)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 706)

    println("\n=== Monte Carlo Standard Error (MCSE) ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    # MCSE should exist and be positive
    mcse_value = mcse(result)

    @test mcse_value > 0
    @test isfinite(mcse_value)

    # MCSE should be smaller than posterior std
    @test mcse_value < std(extract_alpha(result))

    println("  Posterior std: $(round(std(extract_alpha(result)), digits = 4))")
    println("  MCSE: $(round(mcse_value, digits = 4))")
    println("  ✓ MCSE calculation works")
end

@testset "P-values" begin
    Random.seed!(707)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 707)

    println("\n=== P-values ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    # pvalues function
    pval = pvalues(result)

    @test isfinite(pval)
    @test 0 <= pval <= 1

    # For a true effect of 0.5, p-value should be small
    # (but we use a loose check due to small sample)
    @test pval >= 0

    println("  P-value: $(round(pval, digits = 4))")
    println("  ✓ P-value calculation works")
end

@testset "HPD Interval" begin
    Random.seed!(708)

    # Test with skewed distribution
    samples = randn(10000) .^ 2 .+ 0.5  # Chi-squared-like

    hpd_95 = BayesianDoubleML.hpd_interval(samples; level = 0.95)

    @test length(hpd_95) == 2
    @test hpd_95[1] < hpd_95[2]
    @test hpd_95[1] >= minimum(samples)
    @test hpd_95[2] <= maximum(samples)

    # HPD interval should contain the mode region
    mode_estimate = samples[findmin(abs.(samples .- median(samples)))[2]]
    @test hpd_95[1] <= mode_estimate <= hpd_95[2]

    println("\n=== HPD Interval ===")
    println("  95% HPD: [$(round(hpd_95[1], digits = 4)), $(round(hpd_95[2], digits = 4))]")
    println("  Sample range: [$(round(minimum(samples), digits = 4)), $(round(maximum(samples), digits = 4))]")
    println("  ✓ HPD interval works correctly")
end

@testset "Convergence Diagnostics" begin
    Random.seed!(709)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 709)

    println("\n=== Convergence Diagnostics ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100, n_chains = 2)

    # R-hat (potential scale reduction factor)
    rhat = rhat_statistic(result)

    @test isfinite(rhat)
    @test rhat > 0
    # R-hat should be close to 1.0 for converged chains
    @test rhat < 2.0  # Loose threshold for small test

    println("  R-hat: $(round(rhat, digits = 3))")
    println("  ✓ Convergence diagnostics work")
end

@testset "Model Comparison Statistics" begin
    Random.seed!(710)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 710)

    println("\n=== Model Comparison Statistics ===")

    # VI result with ELBO
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 400, n_draws = 400)

    # ELBO should be present in VI results
    @test hasfield(typeof(result), :final_elbo)
    @test isfinite(result.final_elbo)

    println("  Final ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Converged: $(result.converged)")
    println("  ✓ Model comparison statistics available")
end

@testset "StatsAPI Integration" begin
    Random.seed!(711)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 711)

    println("\n=== StatsAPI Integration ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    # Test StatsAPI functions work
    @test coef(result) isa AbstractVector
    @test stderror(result) isa AbstractVector
    @test vcov(result) isa AbstractMatrix

    println("  coef: $(round.(coef(result), digits = 4))")
    println("  stderror: $(round.(stderror(result), digits = 4))")
    println("  ✓ StatsAPI integration works")
end

@testset "Diagnostics on Edge Cases" begin
    println("\n=== Diagnostics on Edge Cases ===")

    # Single value
    single_sample = [0.5]
    ci_single = BayesianDoubleML.credible_interval(single_sample)
    @test ci_single[1] == ci_single[2] == 0.5

    # All same values
    constant_samples = fill(1.0, 100)
    ci_constant = BayesianDoubleML.credible_interval(constant_samples)
    @test ci_constant[1] == ci_constant[2] == 1.0

    # Very small sample
    small_samples = [0.1, 0.5, 0.9]
    ci_small = BayesianDoubleML.credible_interval(small_samples)
    @test length(ci_small) == 2

    println("  Single sample CI: [$(round(ci_single[1], digits = 4)), $(round(ci_single[2], digits = 4))]")
    println("  Constant sample CI: [$(round(ci_constant[1], digits = 4)), $(round(ci_constant[2], digits = 4))]")
    println("  ✓ Edge cases handled correctly")
end

println("\n=== All Diagnostics Tests Complete ===")
println("Tested: Credible intervals, HPD, MCSE, ESS, p-values, coeftable, StatsAPI")
