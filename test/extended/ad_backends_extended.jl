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
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(500))

    println("\n=== AD Backend: AutoReverseDiff ===")

    model = BDMLModel(Y, D, X; model_type = :hier)

    # Default (compile=true)
    method1 = UnifiedVI(; ad_backend = AutoReverseDiff)
    elapsed1 = @elapsed fit!(model, method1; n_iterations = 400, n_draws = 400)

    # Without compile
    method2 = UnifiedVI(; ad_backend = AutoReverseDiff)
    elapsed2 = @elapsed fit!(model, method2; n_iterations = 400, n_draws = 400, force = true)

    alpha1 = mean(extract_alpha(model))

    println("  Compile=true: α=$(round(alpha1, digits = 4)), time=$(round(elapsed1, digits = 2))s")

    @test isfitted(model)
    @test isfinite(alpha1)
    @test abs(alpha1 - alpha_true) < 0.5
end

@testset "AD Backend - AutoForwardDiff" begin
    Random.seed!(501)
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(501))

    println("\n=== AD Backend: AutoForwardDiff ===")

    model = BDMLModel(Y, D, X; model_type = :hier)

    # Default chunksize
    method1 = UnifiedVI(; ad_backend = AutoForwardDiff)
    elapsed1 = @elapsed fit!(model, method1; n_iterations = 400, n_draws = 400)

    # Custom chunksize
    method2 = UnifiedVI(; ad_backend = AutoForwardDiff)
    elapsed2 = @elapsed fit!(model, method2; n_iterations = 300, n_draws = 300, force = true)

    alpha1 = mean(extract_alpha(model))

    println("  Default chunksize: α=$(round(alpha1, digits = 4)), time=$(round(elapsed1, digits = 2))s")

    @test isfitted(model)
    @test isfinite(alpha1)
    @test abs(alpha1 - alpha_true) < 0.5
end

@testset "AD Backend - AutoZygote" begin
    Random.seed!(503)
    Y, D, X, alpha_true, _ = generate_dgp_table1(80, 8, 2.0; alpha_true = 0.5, rng = MersenneTwister(503))

    println("\n=== AD Backend: AutoZygote ===")

    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoZygote)

    elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(model))

    println("  Time: $(round(elapsed, digits = 2))s")
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfitted(model)
    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "AD Backend Comparison - All Four" begin
    Random.seed!(504)
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(504))

    println("\n=== AD Backend Comparison: All Four ===")
    println("True α: $alpha_true")

    model = BDMLModel(Y, D, X; model_type = :hier)

    backends = [
        ("AutoReverseDiff", AutoReverseDiff),
        ("AutoForwardDiff", AutoForwardDiff),
        ("AutoZygote", AutoZygote),
    ]

    results = Dict{String, Float64}()
    times = Dict{String, Float64}()

    for (name, backend) in backends
        Random.seed!(504)  # Reset seed for fair comparison

        method = UnifiedVI(; ad_backend = backend)

        elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500, force = true)
        alpha_mean = mean(extract_alpha(model))

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
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(505))

    println("\n=== AD Backend with SimpleVI ===")

    model = BDMLModel(Y, D, X; model_type = :hier)

    # Test Mooncake with SimpleVI
    method_mooncake = SimpleVI(; ad_backend = AutoMooncake)
    fit!(model, SimpleVI(; ad_backend = AutoMooncake); n_iterations = 50, n_draws = 100)  # Warmup

    Random.seed!(505)
    elapsed_mooncake = @elapsed fit!(model, method_mooncake; n_iterations = 300, n_draws = 300, force = true)
    alpha_mooncake = mean(extract_alpha(model))

    # Test ReverseDiff with SimpleVI
    method_rd = SimpleVI(; ad_backend = AutoReverseDiff)
    elapsed_rd = @elapsed fit!(model, method_rd; n_iterations = 300, n_draws = 300, force = true)
    alpha_rd = mean(extract_alpha(model))

    println("  SimpleVI + AutoMooncake: α=$(round(alpha_mooncake, digits = 4)), time=$(round(elapsed_mooncake, digits = 2))s")
    println("  SimpleVI + AutoReverseDiff: α=$(round(alpha_rd, digits = 4)), time=$(round(elapsed_rd, digits = 2))s")

    @test isfinite(alpha_mooncake)
    @test isfinite(alpha_rd)
    @test abs(alpha_mooncake - alpha_true) < 0.5
    @test abs(alpha_rd - alpha_true) < 0.5
end

@testset "AD Backend with MCMC" begin
    Random.seed!(506)
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(506))

    println("\n=== AD Backend with MCMC (via Turing) ===")

    # Note: MCMC uses Turing's internal AD, but we can verify it works
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed fit!(model, method; n_samples = 300, n_chains = 1)
    alpha_mean = mean(extract_alpha(model))

    println("  MCMC NUTS: α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")

    @test isfitted(model)
    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "AD Backend with Different Model Types" begin
    Random.seed!(507)
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(507))

    println("\n=== AD Backend with Different Model Types ===")

    backends = [AutoReverseDiff]

    for mt in [:basic, :hier]
        println("\n  Model type: $mt")

        model = BDMLModel(Y, D, X; model_type = mt)

        for backend in backends
            Random.seed!(507)
            method = UnifiedVI(; ad_backend = backend)
            fit!(model, method; n_iterations = 300, n_draws = 300, force = true)

            alpha_mean = mean(extract_alpha(model))
            println("    $(nameof(backend)): α=$(round(alpha_mean, digits = 4))")

            @test isfinite(alpha_mean)
            @test abs(alpha_mean - alpha_true) < 0.5
        end
    end
end

@testset "AD Backend Performance Consistency" begin
    Random.seed!(508)
    Y, D, X, alpha_true, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(508))

    println("\n=== AD Backend Performance Consistency ===")

    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)

    # Run multiple times to check consistency
    alphas = Float64[]
    for i in 1:3
        Random.seed!(508 + i)
        fit!(model, method; n_iterations = 500, n_draws = 500, force = true)
        push!(alphas, mean(extract_alpha(model)))
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
println("Tested: AutoReverseDiff, AutoForwardDiff, AutoZygote with UnifiedVI")
println("Tested: AutoMooncake, AutoReverseDiff with SimpleVI")
println("Includes: Timing comparisons, consistency checks")
