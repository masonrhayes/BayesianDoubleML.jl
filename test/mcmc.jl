# MCMC Inference Tests
# Tests for fit() with MCMCMethod, NUTS and HMC samplers
# Single and multi-chain inference with both basic and hierarchical models

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "MCMC Single Chain - Basic Model" begin
    Random.seed!(200)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 200)

    println("\n=== MCMC Single Chain: Basic Model ===")
    println("True α: $alpha_true")

    # Create problem and fit with NUTS
    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = MCMCNUTS()  # Method only takes algorithm params

    elapsed = @elapsed result = fit(problem, method; n_samples = 200, n_chains = 1)

    @test result isa BDMLMCMCResult
    @test result.model_type == :basic
    @test length(result.alpha_samples) >= 200

    alpha_mean = mean(extract_alpha(result))
    alpha_std = std(extract_alpha(result))
    ci = credible_interval(result)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "MCMC Single Chain - Hierarchical Model" begin
    Random.seed!(201)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.8, seed = 201)

    println("\n=== MCMC Single Chain: Hierarchical Model ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_samples = 200, n_chains = 1)

    @test result isa BDMLMCMCResult
    @test result.model_type == :hier
    @test length(result.alpha_samples) >= 200

    alpha_mean = mean(extract_alpha(result))
    alpha_std = std(extract_alpha(result))
    ci = credible_interval(result)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "MCMC Multi-Chain - Hierarchical Model" begin
    Random.seed!(202)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.6, seed = 202)

    println("\n=== MCMC Multi-Chain: Hierarchical Model ===")
    println("True α: $alpha_true")
    println("Chains: 2")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_samples = 150, n_chains = 2)

    alpha_mean = mean(extract_alpha(result))
    alpha_std = std(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  Total samples: $(length(result.alpha_samples))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "MCMC Binary Treatment Data - IRM" begin
    Random.seed!(205)
    Y, D, X, alpha_true = make_binary_treatment_data(n = 100, p = 10, alpha_true = 0.5, seed = 205)

    println("\n=== MCMC Binary Treatment: Hierarchical Model ===")
    println("True α: $alpha_true")
    println("Treatment type: Binary (IRM)")

    @test all(D .∈ Ref([0.0, 1.0]))

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_samples = 200, n_chains = 1)

    @test result isa BDMLMCMCResult

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "MCMC BDMLData Input" begin
    Random.seed!(206)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 206)

    println("\n=== MCMC with BDMLData Input ===")

    data = BDMLData(Y, D, X)
    problem = BDMLProblem(data; model_type = :hier)
    method = MCMCNUTS()

    result = fit(problem, method; n_samples = 200, n_chains = 1)

    @test result isa BDMLMCMCResult
    @test length(result.alpha_samples) >= 200

    alpha_mean = mean(extract_alpha(result))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "MCMC Result Display and Summary" begin
    Random.seed!(207)
    Y, D, X, alpha_true = make_test_data(n = 50, p = 5, alpha_true = 0.5, seed = 207)

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    result = fit(problem, method; n_samples = 200, n_chains = 1)

    # Test display methods
    @test_nowarn println(result)
    @test_nowarn show(result)

    # Test credible_interval on result
    ci = credible_interval(result)
    @test length(ci) == 2
    @test ci[1] < ci[2]

    alpha_mean = mean(extract_alpha(result))
    @test ci[1] < alpha_mean < ci[2]
end

@testset "MCMC Alpha Extraction Validation" begin
    Random.seed!(208)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 208)

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    result = fit(problem, method; n_samples = 200, n_chains = 1)

    alpha_samples = result.alpha_samples

    # All samples should be finite
    @test all(isfinite.(alpha_samples))

    # Reasonable range check
    @test minimum(alpha_samples) > -5
    @test maximum(alpha_samples) < 5

    # Mean should not be extreme
    alpha_mean = mean(alpha_samples)
    @test -3 < alpha_mean < 3
end

@testset "MCMC Target Acceptance Rate Variations" begin
    Random.seed!(209)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 209)

    println("\n=== MCMC Target Acceptance Rate Variations ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Test with different target acceptance rates
    for target_acc in [0.65, 0.8, 0.95]
        method = MCMCNUTS(; target_acceptance = target_acc)
        result = fit(problem, method; n_samples = 100, n_chains = 1)

        alpha_mean = mean(extract_alpha(result))
        println("  Target $target_acc: α = $(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
    end
end

@testset "MCMC Max Depth Variations" begin
    Random.seed!(210)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 210)

    println("\n=== MCMC Max Depth Variations ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Test with different max depths
    for max_d in [5, 10, 15]
        method = MCMCNUTS(; max_depth = max_d)
        result = fit(problem, method; n_samples = 100, n_chains = 1)

        alpha_mean = mean(extract_alpha(result))
        println("  Max depth $max_d: α = $(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
    end
end

println("\n=== All MCMC Tests Complete ===")
println("Tested: NUTS single/multi-chain, basic/hierarchical models")
