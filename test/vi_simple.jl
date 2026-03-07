# Simple VI Tests
# Tests for fit() with SimpleVIMethod/SimpleVI
# Lightweight VI with Mooncake default

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "SimpleVI Basic Model" begin
    Random.seed!(400)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 400)

    println("\n=== SimpleVI: Basic Model ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = SimpleVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :basic
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI Hierarchical Model" begin
    Random.seed!(401)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.8, seed = 401)

    println("\n=== SimpleVI: Hierarchical Model ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :hier
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI with BDMLData" begin
    Random.seed!(402)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 402)

    println("\n=== SimpleVI with BDMLData ===")

    data = BDMLData(Y, D, X)
    problem = BDMLProblem(data; model_type = :hier)
    method = SimpleVI()

    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "SimpleVI AD Backend - AutoMooncake" begin
    Random.seed!(403)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 403)

    println("\n=== SimpleVI: AutoMooncake Backend ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI(; ad_backend = AutoMooncake)

    # Warmup
    println("  Warmup...")
    _ = fit(problem, SimpleVI(; ad_backend = AutoMooncake); n_iterations = 50, n_draws = 100)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI AD Backend - AutoReverseDiff" begin
    Random.seed!(404)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 404)

    println("\n=== SimpleVI: AutoReverseDiff Backend ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI(; ad_backend = AutoReverseDiff)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI Binary Treatment" begin
    Random.seed!(405)
    Y, D, X, alpha_true = make_binary_treatment_data(n = 100, p = 10, alpha_true = 0.5, seed = 405)

    println("\n=== SimpleVI: Binary Treatment ===")
    println("True α: $alpha_true")

    @test all(D .∈ Ref([0.0, 1.0]))

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()

    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(result.alpha_samples)
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI Small Dataset" begin
    Random.seed!(406)
    Y, D, X, alpha_true = make_test_data(n = 50, p = 5, alpha_true = 0.5, seed = 406)

    println("\n=== SimpleVI: Small Dataset (n=50, p=5) ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()

    result = fit(problem, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "SimpleVI Result Display" begin
    Random.seed!(407)
    Y, D, X, alpha_true = make_test_data(n = 50, p = 5, alpha_true = 0.5, seed = 407)

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()

    result = fit(problem, method; n_iterations = 300, n_draws = 300)

    @test_nowarn println(result)
    @test_nowarn show(result)

    ci = credible_interval(result)
    @test length(ci) == 2

    alpha_mean = mean(result.alpha_samples)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "SimpleVI vs UnifiedVI Comparison" begin
    Random.seed!(408)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 408)

    println("\n=== SimpleVI vs UnifiedVI Comparison ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # SimpleVI
    method_simple = SimpleVI()
    result_simple = fit(problem, method_simple)
    alpha_simple = mean(result_simple.alpha_samples)

    # UnifiedVI
    method_unified = UnifiedVI()
    result_unified = fit(problem, method_unified)
    alpha_unified = mean(result_unified.alpha_samples)

    println("  SimpleVI: α = $(round(alpha_simple, digits = 4))")
    println("  UnifiedVI: α = $(round(alpha_unified, digits = 4))")
    println("  Difference: $(round(abs(alpha_simple - alpha_unified), digits = 4))")

    # Both should be close to true value
    @test abs(alpha_simple - alpha_true) < 0.5
    @test abs(alpha_unified - alpha_true) < 0.5

    # Results should be reasonably similar
    @test abs(alpha_simple - alpha_unified) < 0.5
end

@testset "SimpleVI Different Iterations" begin
    Random.seed!(409)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 409)

    println("\n=== SimpleVI: Different Iteration Counts ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    for n_iter in [200, 500, 800]
        method = SimpleVI()
        result = fit(problem, method; n_iterations = n_iter, n_draws = 400)

        alpha_mean = mean(result.alpha_samples)
        println("  n_iterations=$n_iter: α = $(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
    end
end

println("\n=== All SimpleVI Tests Complete ===")
println("Tested: Basic/hierarchical models, AD backends, binary treatment")
