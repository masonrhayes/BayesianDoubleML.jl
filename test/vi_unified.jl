# Unified VI Tests
# Tests for fit() with UnifiedVIMethod/UnifiedVI
# Basic and hierarchical models with various configurations

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "UnifiedVI Basic Model" begin
    Random.seed!(300)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 300)

    println("\n=== UnifiedVI: Basic Model ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI()

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
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Converged: $(result.converged)")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
    @test isfinite(result.final_elbo)
end

@testset "UnifiedVI Hierarchical Model" begin
    Random.seed!(301)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.8, seed = 301)

    println("\n=== UnifiedVI: Hierarchical Model ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :hier
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI with BDMLData" begin
    Random.seed!(302)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 302)

    println("\n=== UnifiedVI with BDMLData ===")

    data = BDMLData(Y, D, X)
    problem = BDMLProblem(data; model_type = :hier)
    method = UnifiedVI()

    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "UnifiedVI AD Backend - AutoReverseDiff" begin
    Random.seed!(303)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 303)

    println("\n=== UnifiedVI: AutoReverseDiff ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI AD Backend - AutoForwardDiff" begin
    Random.seed!(304)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 304)

    println("\n=== UnifiedVI: AutoForwardDiff ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoForwardDiff)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI AD Backend - AutoMooncake" begin
    Random.seed!(305)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 305)

    println("\n=== UnifiedVI: AutoMooncake ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoMooncake)

    # Warmup for compilation
    println("  Warmup...")
    _ = fit(problem, UnifiedVI(; ad_backend = AutoMooncake); n_iterations = 50, n_draws = 100)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI AD Backend - AutoZygote" begin
    Random.seed!(306)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 306)

    println("\n=== UnifiedVI: AutoZygote ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoZygote)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "UnifiedVI with Subsampling Enabled" begin
    Random.seed!(307)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 307)

    println("\n=== UnifiedVI with Subsampling ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; subsample = true, batch_size = 64)

    result = fit(problem, method; n_iterations = 400, n_draws = 400)

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "UnifiedVI Monte Carlo Samples Variation" begin
    Random.seed!(308)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 308)

    println("\n=== UnifiedVI: n_montecarlo Variation ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    for n_mc in [5, 10, 20]
        method = UnifiedVI(; n_montecarlo = n_mc)
        result = fit(problem, method; n_iterations = 300, n_draws = 300)

        alpha_mean = mean(result.alpha_samples)
        println("  n_montecarlo=$n_mc: α = $(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
    end
end

@testset "UnifiedVI Binary Treatment" begin
    Random.seed!(309)
    Y, D, X, alpha_true = make_binary_treatment_data(n = 100, p = 10, alpha_true = 0.5, seed = 309)

    println("\n=== UnifiedVI: Binary Treatment (IRM) ===")
    println("True α: $alpha_true")

    @test all(D .∈ Ref([0.0, 1.0]))

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(result.alpha_samples)
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI Result Display and Credible Interval" begin
    Random.seed!(310)
    Y, D, X, alpha_true = make_test_data(n = 50, p = 5, alpha_true = 0.5, seed = 310)

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    result = fit(problem, method; n_iterations = 300, n_draws = 300)

    # Test display methods
    @test_nowarn println(result)
    @test_nowarn show(result)

    # Test credible interval
    ci = credible_interval(result)
    @test length(ci) == 2
    @test ci[1] < ci[2]

    alpha_mean = mean(result.alpha_samples)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "UnifiedVI Convergence Detection" begin
    Random.seed!(311)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 311)

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    result = fit(problem, method; n_iterations = 1000, n_draws = 500)

    @test hasfield(typeof(result), :converged)
    @test hasfield(typeof(result), :final_elbo)
    @test isfinite(result.final_elbo)

    println("  Converged: $(result.converged)")
    println("  Final ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Iterations: $(result.n_iterations)")
end

println("\n=== All UnifiedVI Tests Complete ===")
println("Tested: Basic/hierarchical models, multiple AD backends, subsampling")
