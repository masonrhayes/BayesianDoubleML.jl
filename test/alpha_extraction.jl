# Alpha Extraction Tests
# Tests for all alpha extraction methods from MCMC and VI results
# Correct parameter indexing and extraction validation

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "Alpha Extraction from MCMC - Hierarchical Model" begin
    Random.seed!(800)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 800)

    println("\n=== Alpha Extraction: MCMC Hierarchical ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    # Check alpha_samples field exists
    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 200

    # Check all samples are finite
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  All samples finite: $(all(isfinite.(result.alpha_samples)))")
    println("  ✓ Alpha extraction from MCMC hierarchical works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from MCMC - Basic Model" begin
    Random.seed!(801)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.8, seed = 801)

    println("\n=== Alpha Extraction: MCMC Basic ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 200
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from MCMC basic works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from VI - Hierarchical Model" begin
    Random.seed!(802)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 802)

    println("\n=== Alpha Extraction: VI Hierarchical ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 500
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI hierarchical works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from VI - Basic Model" begin
    Random.seed!(803)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.6, seed = 803)

    println("\n=== Alpha Extraction: VI Basic ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 500
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI basic works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from SimpleVI" begin
    Random.seed!(804)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 804)

    println("\n=== Alpha Extraction: SimpleVI ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()
    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 500
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from SimpleVI works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Range Validation" begin
    Random.seed!(805)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 805)

    println("\n=== Alpha Range Validation ===")

    # Test MCMC
    problem_mcmc = BDMLProblem(Y, D, X; model_type = :hier)
    result_mcmc = fit(problem_mcmc, MCMCNUTS(; n_samples = 200, n_warmup = 100))

    @test minimum(result_mcmc.alpha_samples) > -10
    @test maximum(result_mcmc.alpha_samples) < 10

    # Test VI
    problem_vi = BDMLProblem(Y, D, X; model_type = :hier)
    result_vi = fit(problem_vi, UnifiedVI())

    @test minimum(result_vi.alpha_samples) > -10
    @test maximum(result_vi.alpha_samples) < 10

    println("  MCMC α range: [$(round(minimum(result_mcmc.alpha_samples), digits = 4)), $(round(maximum(result_mcmc.alpha_samples), digits = 4))]")
    println("  VI α range: [$(round(minimum(result_vi.alpha_samples), digits = 4)), $(round(maximum(result_vi.alpha_samples), digits = 4))]")
    println("  ✓ Alpha values in reasonable range")
end

@testset "Alpha Statistics Consistency" begin
    Random.seed!(806)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 806)

    println("\n=== Alpha Statistics Consistency ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    # Manual calculation should match result.mean_alpha if available
    alpha_manual_mean = mean(extract_alpha(result))
    alpha_manual_std = std(extract_alpha(result))

    # Basic sanity checks
    @test alpha_manual_mean > -5
    @test alpha_manual_mean < 5
    @test alpha_manual_std > 0

    # CI should contain mean
    ci = credible_interval(result)
    @test ci[1] < alpha_manual_mean < ci[2]

    println("  Mean: $(round(alpha_manual_mean, digits = 4))")
    println("  Std: $(round(alpha_manual_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ✓ Alpha statistics consistent")
end

@testset "Alpha Extraction with Binary Treatment" begin
    Random.seed!(807)
    Y, D, X, alpha_true = make_binary_treatment_data(n = 100, p = 10, alpha_true = 0.5, seed = 807)

    println("\n=== Alpha Extraction: Binary Treatment ===")
    println("True α: $alpha_true")

    @test all(D .∈ Ref([0.0, 1.0]))

    # MCMC
    problem_mcmc = BDMLProblem(Y, D, X; model_type = :hier)
    result_mcmc = fit(problem_mcmc, MCMCNUTS(; n_samples = 200, n_warmup = 100))

    alpha_mcmc = mean(result_mcmc.alpha_samples)

    # VI
    problem_vi = BDMLProblem(Y, D, X; model_type = :hier)
    result_vi = fit(problem_vi, UnifiedVI())

    alpha_vi = mean(result_vi.alpha_samples)

    println("  MCMC α: $(round(alpha_mcmc, digits = 4))")
    println("  VI α: $(round(alpha_vi, digits = 4))")
    println("  ✓ Alpha extraction with binary treatment works")

    @test isfinite(alpha_mcmc)
    @test isfinite(alpha_vi)
end

@testset "Alpha Extraction with Different Seeds" begin
    Random.seed!(808)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 808)

    println("\n=== Alpha Extraction: Different Seeds ===")

    alphas = Float64[]

    for seed in [100, 200, 300]
        Random.seed!(seed)
        problem = BDMLProblem(Y, D, X; model_type = :hier)
        method = UnifiedVI()
        result = fit(problem, method; n_iterations = 400, n_draws = 400)

        alpha_mean = mean(extract_alpha(result))
        push!(alphas, alpha_mean)

        println("  Seed $seed: α = $(round(alpha_mean, digits = 4))")
    end

    # Results should be in reasonable range
    @test all(isfinite.(alphas))
    @test all(abs.(alphas .- alpha_true) .< 0.5)

    # Some variation expected, but not too much
    @test maximum(alphas) - minimum(alphas) < 1.0

    println("  Range: $(round(maximum(alphas) - minimum(alphas), digits = 4))")
    println("  ✓ Alpha extraction stable across seeds")
end

@testset "Alpha Extraction with Multi-Chain MCMC" begin
    Random.seed!(809)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 809)

    println("\n=== Alpha Extraction: Multi-Chain MCMC ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 150, n_warmup = 100, n_chains = 2)

    @test length(result.alpha_samples) >= 300  # 150 * 2
    @test all(isfinite.(result.alpha_samples))

    alpha_mean = mean(extract_alpha(result))

    println("  Chains: 2")
    println("  Total samples: $(length(result.alpha_samples))")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from multi-chain works")

    @test isfinite(alpha_mean)
end

@testset "Alpha Posterior Distribution Shape" begin
    Random.seed!(810)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 810)

    println("\n=== Alpha Posterior Distribution Shape ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 200)

    samples = result.alpha_samples

    # Distribution properties
    alpha_mean = mean(samples)
    alpha_std = std(samples)
    alpha_median = median(samples)

    # Should be roughly normal/symmetric for this DGP
    skewness_approx = mean(((samples .- alpha_mean) ./ alpha_std) .^ 3)

    println("  Mean: $(round(alpha_mean, digits = 4))")
    println("  Median: $(round(alpha_median, digits = 4))")
    println("  Std: $(round(alpha_std, digits = 4))")
    println("  Approx skewness: $(round(skewness_approx, digits = 4))")
    println("  ✓ Posterior distribution has reasonable shape")

    # Mean and median should be close for symmetric distributions
    @test abs(alpha_mean - alpha_median) < alpha_std

    # Skewness should not be extreme
    @test abs(skewness_approx) < 2.0
end

@testset "Alpha Coefficient Table Extraction" begin
    Random.seed!(811)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 811)

    println("\n=== Alpha from Coefficient Table ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    ct = coeftable(result)

    # Find alpha in coefficient table
    alpha_idx = findfirst(contains("alpha"), ct.rownms)
    @test alpha_idx !== nothing

    alpha_from_ct = ct.mat[alpha_idx, 1]
    alpha_from_samples = mean(extract_alpha(result))

    println("  From samples: $(round(alpha_from_samples, digits = 4))")
    println("  From coeftable: $(round(alpha_from_ct, digits = 4))")
    println("  ✓ Alpha accessible from coefficient table")

    # These should match closely
    @test abs(alpha_from_ct - alpha_from_samples) < 0.01
end

@testset "Alpha from MCMC Result" begin
    Random.seed!(812)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 812)

    println("\n=== Alpha from MCMC Result ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    result = fit(problem, MCMCNUTS(); n_samples = 200, n_warmup = 100)

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from MCMC works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha from VI Result" begin
    Random.seed!(813)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 813)

    println("\n=== Alpha from VI Result ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    result = fit(
        problem, UnifiedVIMethod();
        n_iterations = 500,
        n_draws = 500,
        show_progress = false
    )

    @test hasfield(typeof(result), :alpha_samples)
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

println("\n=== All Alpha Extraction Tests Complete ===")
println("Tested: MCMC, VI, SimpleVI, binary treatment, multi-chain")
