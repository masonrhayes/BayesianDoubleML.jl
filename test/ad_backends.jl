# AD Backends Tests
# Comprehensive tests for all AD backends with timing comparisons
# AutoReverseDiff, AutoForwardDiff, AutoZygote, AutoMooncake

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "AD Backend - AutoReverseDiff (Default)" begin
    Random.seed!(500)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 500)

    println("\n=== AD Backend: AutoReverseDiff ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Default (compile=true)
    method1 = UnifiedVI(; ad_backend = AutoReverseDiff)
    elapsed1 = @elapsed result1 = fit(problem, method1; n_iterations = 400, n_draws = 400)

    # Without compile
    method2 = UnifiedVI(; ad_backend = AutoReverseDiff, ad_kwargs = (compile = false,), n_iterations = 400, n_draws = 400)
    elapsed2 = @elapsed result2 = fit(problem, method2; n_iterations = 400, n_draws = 400)

    alpha1 = mean(extract_alpha(result1))
    alpha2 = mean(extract_alpha(result2))

    println("  Compile=true: α=$(round(alpha1, digits = 4)), time=$(round(elapsed1, digits = 2))s")
    println("  Compile=false: α=$(round(alpha2, digits = 4)), time=$(round(elapsed2, digits = 2))s")

    @test isfinite(alpha1)
    @test isfinite(alpha2)
    @test abs(alpha1 - alpha_true) < 0.5
    @test abs(alpha2 - alpha_true) < 0.5
    @test abs(alpha1 - alpha2) < 0.3
end

@testset "AD Backend - AutoForwardDiff" begin
    Random.seed!(501)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 501)

    println("\n=== AD Backend: AutoForwardDiff ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Default chunksize
    method1 = UnifiedVI(; ad_backend = AutoForwardDiff)
    elapsed1 = @elapsed result1 = fit(problem, method1; n_iterations = 400, n_draws = 400)

    # Custom chunksize
    method2 = UnifiedVI(; ad_backend = AutoForwardDiff, ad_kwargs = (chunksize = 10,), n_iterations = 300, n_draws = 300)
    elapsed2 = @elapsed result2 = fit(problem, method2; n_iterations = 400, n_draws = 400)

    alpha1 = mean(extract_alpha(result1))
    alpha2 = mean(extract_alpha(result2))

    println("  Default chunksize: α=$(round(alpha1, digits = 4)), time=$(round(elapsed1, digits = 2))s")
    println("  chunksize=10: α=$(round(alpha2, digits = 4)), time=$(round(elapsed2, digits = 2))s")

    @test isfinite(alpha1)
    @test isfinite(alpha2)
    @test abs(alpha1 - alpha_true) < 0.5
    @test abs(alpha2 - alpha_true) < 0.5
end

@testset "AD Backend - AutoMooncake" begin
    Random.seed!(502)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 502)

    println("\n=== AD Backend: AutoMooncake ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Warmup run
    println("  Warmup run...")
    method_warmup = UnifiedVI(; ad_backend = AutoMooncake)
    _ = fit(problem, method_warmup; n_iterations = 50, n_draws = 100)

    # Actual timing run
    method = UnifiedVI(; ad_backend = AutoMooncake)
    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Time: $(round(elapsed, digits = 2))s")
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "AD Backend - AutoZygote" begin
    Random.seed!(503)
    Y, D, X, alpha_true = make_test_data(n = 80, p = 8, alpha_true = 0.5, seed = 503)

    println("\n=== AD Backend: AutoZygote ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoZygote)

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Time: $(round(elapsed, digits = 2))s")
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "AD Backend Comparison - All Four" begin
    Random.seed!(504)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 504)

    println("\n=== AD Backend Comparison: All Four ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    backends = [
        ("AutoReverseDiff", AutoReverseDiff),
        ("AutoForwardDiff", AutoForwardDiff),
        ("AutoMooncake", AutoMooncake),
        ("AutoZygote", AutoZygote),
    ]

    results = Dict{String, Float64}()
    times = Dict{String, Float64}()

    for (name, backend) in backends
        Random.seed!(504)  # Reset seed for fair comparison

        method = UnifiedVI(; ad_backend = backend)

        # Warmup for Mooncake
        if backend == AutoMooncake
            _ = fit(problem, UnifiedVI(; ad_backend = AutoMooncake))
            Random.seed!(504)
        end

        elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)
        alpha_mean = mean(extract_alpha(result))

        results[name] = alpha_mean
        times[name] = elapsed

        println("  $name: α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")
    end

    println("\n  --- Results Summary ---")
    println("  Estimates range: [$(round(minimum(values(results)), digits = 4)), $(round(maximum(values(results)), digits = 4))]")
    println("  Time range: [$(round(minimum(values(times)), digits = 2))s, $(round(maximum(values(times)), digits = 2))s]")

    # All backends should produce similar results
    for (name, alpha) in results
        @test isfinite(alpha)
        @test abs(alpha - alpha_true) < 0.5
    end

    # Max difference between backends should be reasonable
    alpha_values = collect(values(results))
    @test maximum(alpha_values) - minimum(alpha_values) < 0.5
end

@testset "AD Backend with SimpleVI" begin
    Random.seed!(505)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 505)

    println("\n=== AD Backend with SimpleVI ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Test Mooncake with SimpleVI
    method_mooncake = SimpleVI(; ad_backend = AutoMooncake)
    _ = fit(problem, SimpleVI(; ad_backend = AutoMooncake))  # Warmup

    Random.seed!(505)
    elapsed_mooncake = @elapsed result_mooncake = fit(problem, method_mooncake; n_iterations = 300, n_draws = 300)
    alpha_mooncake = mean(result_mooncake.alpha_samples)

    # Test ReverseDiff with SimpleVI
    method_rd = SimpleVI(; ad_backend = AutoReverseDiff)
    elapsed_rd = @elapsed result_rd = fit(problem, method_rd; n_iterations = 300, n_draws = 300)
    alpha_rd = mean(result_rd.alpha_samples)

    println("  SimpleVI + AutoMooncake: α=$(round(alpha_mooncake, digits = 4)), time=$(round(elapsed_mooncake, digits = 2))s")
    println("  SimpleVI + AutoReverseDiff: α=$(round(alpha_rd, digits = 4)), time=$(round(elapsed_rd, digits = 2))s")

    @test isfinite(alpha_mooncake)
    @test isfinite(alpha_rd)
    @test abs(alpha_mooncake - alpha_true) < 0.5
    @test abs(alpha_rd - alpha_true) < 0.5
end

@testset "AD Backend with MCMC" begin
    Random.seed!(506)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 506)

    println("\n=== AD Backend with MCMC (via Turing) ===")

    # Note: MCMC uses Turing's internal AD, but we can verify it works
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)
    alpha_mean = mean(extract_alpha(result))

    println("  MCMC NUTS: α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "AD Backend with Different Model Types" begin
    Random.seed!(507)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 507)

    println("\n=== AD Backend with Different Model Types ===")

    backends = [AutoReverseDiff, AutoMooncake]

    for model_type in [:basic, :hier]
        println("\n  Model type: $model_type")

        for backend in backends
            problem = BDMLProblem(Y, D, X; model_type = model_type)

            # Warmup for Mooncake
            if backend == AutoMooncake
                _ = fit(problem, UnifiedVI(; ad_backend = AutoMooncake))
            end

            Random.seed!(507)
            method = UnifiedVI(; ad_backend = backend)
            result = fit(problem, method; n_iterations = 300, n_draws = 300)

            alpha_mean = mean(extract_alpha(result))
            println("    $(nameof(backend)): α=$(round(alpha_mean, digits = 4))")

            @test isfinite(alpha_mean)
            @test abs(alpha_mean - alpha_true) < 0.5
        end
    end
end

@testset "AD Backend Performance Consistency" begin
    Random.seed!(508)
    Y, D, X, alpha_true = make_test_data(n = 100, p = 10, alpha_true = 0.5, seed = 508)

    println("\n=== AD Backend Performance Consistency ===")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)

    # Run multiple times to check consistency
    alphas = Float64[]
    for i in 1:3
        Random.seed!(508 + i)
        result = fit(problem, method; n_iterations = 500, n_draws = 500)
        push!(alphas, mean(extract_alpha(result)))
    end

    println("  Multiple runs with same seed offset:")
    for (i, alpha) in enumerate(alphas)
        println("    Run $i: α=$(round(alpha, digits = 4))")
    end

    # Check that results are reasonably consistent
    @test all(isfinite.(alphas))
    @test maximum(alphas) - minimum(alphas) < 0.5
    @test all(abs.(alphas .- alpha_true) .< 0.5)
end

println("\n=== All AD Backend Tests Complete ===")
println("Tested: AutoReverseDiff, AutoForwardDiff, AutoMooncake, AutoZygote")
println("Includes: Timing comparisons, compile options, chunksize options, consistency checks")
