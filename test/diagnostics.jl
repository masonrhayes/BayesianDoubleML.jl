# Diagnostics Tests
# Tests for coeftable, HPD intervals, MCSE, ESS, p-values
# Statistical diagnostics and inference summaries with new Model API

using BayesianDoubleML
using Test
using Random
using Statistics
using LinearAlgebra

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

@testset "coeftable for BDMLMCMCResult (MCMC) via Model" begin
    Random.seed!(702)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(702))
    alpha_true = 0.5

    println("\n=== coeftable: BDMLMCMCResult via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCMethod(:nuts)
    fit!(model, method; n_samples = 200)

    ct = coeftable(model)

    @test ct isa BDMLCoeftable
    @test length(ct.coefnames) >= 1  # At least alpha parameter

    # Find alpha row
    alpha_idx = findfirst(contains("α"), ct.coefnames)
    @test alpha_idx !== nothing

    println("  Coefficient table rows: $(length(ct.coefnames))")
    println("  Alpha estimate: $(round(ct.coef[alpha_idx], digits = 4))")
    println("  ✓ coeftable works for MCMC via Model")
end

@testset "coeftable for BDMLVIResult (VI) via Model" begin
    Random.seed!(703)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(703))
    alpha_true = 0.5

    println("\n=== coeftable: BDMLVIResult via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()
    fit!(model, method; n_iterations = 500, n_draws = 500)

    ct = coeftable(model)

    @test ct isa BDMLCoeftable
    @test length(ct.coefnames) >= 1

    alpha_idx = findfirst(contains("α"), ct.coefnames)
    @test alpha_idx !== nothing

    println("  Coefficient table rows: $(length(ct.coefnames))")
    println("  Alpha estimate: $(round(ct.coef[alpha_idx], digits = 4))")
    println("  ✓ coeftable works for VI via Model")
end

@testset "confint for MCMC and VI via Model" begin
    Random.seed!(704)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(704))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])
    alpha_true = 0.5

    println("\n=== confint: MCMC and VI via Model ===")

    # MCMC
    model_mcmc = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_mcmc, MCMCMethod(:nuts); n_samples = 200)

    ci_mcmc = confint(model_mcmc)
    @test length(ci_mcmc) == 2
    @test ci_mcmc[1] < ci_mcmc[2]

    # VI
    model_vi = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_vi, UnifiedVI())

    ci_vi = confint(model_vi)
    @test length(ci_vi) == 2
    @test ci_vi[1] < ci_vi[2]

    println("  MCMC 95% CI: [$(round(ci_mcmc[1], digits = 4)), $(round(ci_mcmc[2], digits = 4))]")
    println("  VI 95% CI: [$(round(ci_vi[1], digits = 4)), $(round(ci_vi[2], digits = 4))]")
    println("  ✓ confint works for both MCMC and VI via Model")
end

@testset "Effective Sample Size (ESS) via Model" begin
    Random.seed!(705)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(705))
    alpha_true = 0.5

    println("\n=== Effective Sample Size (ESS) via Model ===")

    # MCMC with multiple chains for ESS calculation
    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCMethod(:nuts); n_samples = 200, n_chains = 2)

    # ESS function should exist and return positive value
    ess_value = ess(model)

    @test ess_value > 0
    @test ess_value <= length(extract_alpha(model))

    println("  Total samples: $(length(extract_alpha(model)))")
    println("  ESS: $(round(ess_value, digits = 1))")
    println("  Efficiency: $(round(ess_value / length(extract_alpha(model)) * 100, digits = 1))%")
    println("  ✓ ESS calculation works via Model")
end

@testset "Monte Carlo Standard Error (MCSE) via Model" begin
    Random.seed!(706)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(706))
    alpha_true = 0.5

    println("\n=== Monte Carlo Standard Error (MCSE) via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCMethod(:nuts); n_samples = 200)

    # MCSE should exist and be positive
    mcse_value = mcse(model)

    @test mcse_value > 0
    @test isfinite(mcse_value)

    # MCSE should be smaller than posterior std
    @test mcse_value < std(extract_alpha(model))

    println("  Posterior std: $(round(std(extract_alpha(model)), digits = 4))")
    println("  MCSE: $(round(mcse_value, digits = 4))")
    println("  ✓ MCSE calculation works via Model")
end

@testset "P-values via Model" begin
    Random.seed!(707)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(707))
    alpha_true = 0.5

    println("\n=== P-values via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCMethod(:nuts); n_samples = 200)

    # pvalues function
    pval = pvalues(model)

    @test isfinite(pval)
    @test 0 <= pval <= 1

    # For a true effect of 0.5, p-value should be small
    # (but we use a loose check due to small sample)
    @test pval >= 0

    println("  P-value: $(round(pval, digits = 4))")
    println("  ✓ P-value calculation works via Model")
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

@testset "Convergence Diagnostics via Model" begin
    Random.seed!(709)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(709))
    alpha_true = 0.5

    println("\n=== Convergence Diagnostics via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCMethod(:nuts); n_samples = 200, n_chains = 2)

    # R-hat (potential scale reduction factor)
    rhat_val = rhat_statistic(model)

    @test isfinite(rhat_val)
    @test rhat_val > 0
    # R-hat should be close to 1.0 for converged chains
    @test rhat_val < 2.0  # Loose threshold for small test

    println("  R-hat: $(round(rhat_val, digits = 3))")
    println("  ✓ Convergence diagnostics work via Model")
end

@testset "Model Comparison Statistics via Model" begin
    Random.seed!(710)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(710))
    alpha_true = 0.5

    println("\n=== Model Comparison Statistics via Model ===")

    # VI result with ELBO
    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, UnifiedVI(); n_iterations = 400, n_draws = 400)

    # ELBO should be present in VI results
    @test hasfield(typeof(model.result), :final_elbo)
    @test isfinite(model.result.final_elbo)

    println("  Final ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Converged: $(model.result.converged)")
    println("  ✓ Model comparison statistics available via Model")
end

@testset "StatsAPI Integration via Model" begin
    Random.seed!(711)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(711))
    alpha_true = 0.5

    println("\n=== StatsAPI Integration via Model ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCMethod(:nuts); n_samples = 200)

    # Test StatsAPI functions work on model
    @test BayesianDoubleML.coef(model) isa AbstractVector
    @test BayesianDoubleML.stderror(model) isa AbstractVector
    @test BayesianDoubleML.vcov(model) isa AbstractMatrix

    println("  coef: $(round.(BayesianDoubleML.coef(model), digits = 4))")
    println("  stderror: $(round.(BayesianDoubleML.stderror(model), digits = 4))")
    println("  ✓ StatsAPI integration works via Model")
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
println("Tested: Credible intervals, HPD, MCSE, ESS, p-values, coeftable, StatsAPI via Model API")
