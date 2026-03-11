# Subsampling Extended Tests
# Tests for subsampling on large datasets (n>10000)
# Run manually: julia test/extended/subsampling_extended.jl

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include(joinpath(@__DIR__, "..", "utils.jl"))

@testset "Medium Dataset - Auto Subsampling Triggered (n=15000)" begin
    Random.seed!(602)
    n, p = 15000, 20
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Medium Dataset: Auto Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))
    ci = credible_interval(result)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Large Dataset - Explicit Subsampling Control (n=20000)" begin
    Random.seed!(603)
    n, p = 20000, 30
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.8 .* D + X * beta + randn(n)
    alpha_true = 0.8

    println("\n=== Large Dataset: Explicit Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Test with explicit batch size
    method = UnifiedVI(;
        subsample = true,
        batch_size = 128
    )

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Batch size: 128")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Subsampling Batch Size Variations (n=10000)" begin
    Random.seed!(604)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling: Batch Size Variations (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    batch_sizes = [64, 128, 256]

    for bs in batch_sizes
        Random.seed!(604)
        method = UnifiedVI(; subsample = true, batch_size = bs)
        result = fit(problem, method; n_iterations = 300, n_draws = 300)

        alpha_mean = mean(extract_alpha(result))
        println("  Batch size $bs: α=$(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
        @test abs(alpha_mean - alpha_true) < 0.6
    end
end

@testset "Force Full-Batch on Large Dataset (n=12000)" begin
    Random.seed!(605)
    n, p = 12000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Force Full-Batch on Large Dataset (n=$n) ===")
    println("True α: $alpha_true")
    println("Dataset: n=$n (would auto-subsample)")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(;
        subsample = false  # Force full-batch
    )

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "Subsampling with Different AD Backends (n=10000)" begin
    Random.seed!(607)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling with Different AD Backends (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    backends = [AutoReverseDiff]

    for backend in backends
        Random.seed!(607)

        method = UnifiedVI(;
            ad_backend = backend,
            subsample = true,
            batch_size = 128
        )

        elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)
        alpha_mean = mean(extract_alpha(result))

        println("  $(nameof(backend)): α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")

        @test isfinite(alpha_mean)
        @test abs(alpha_mean - alpha_true) < 0.5
    end
end

@testset "Subsampling with Basic Model (n=10000)" begin
    Random.seed!(608)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling with Basic Model (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI(;
        subsample = true,
        batch_size = 128
    )
    result = fit(problem, method; n_iterations = 300, n_draws = 300)
    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Subsampling Accuracy Comparison (n=10000)" begin
    Random.seed!(609)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling Accuracy Comparison (n=$n) ===")
    println("True α: $alpha_true")
    println("Dataset: n=$n")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Full batch (forced)
    Random.seed!(609)
    method_full = UnifiedVI(; subsample = false)
    result_full = fit(problem, method_full; n_iterations = 200, n_draws = 200)
    alpha_full = mean(extract_alpha(result_full))

    # Subsampled
    Random.seed!(609)
    method_sub = UnifiedVI(; subsample = true, batch_size = 128)
    result_sub = fit(problem, method_sub; n_iterations = 300, n_draws = 300)
    alpha_sub = mean(extract_alpha(result_sub))

    println("  Full batch: α=$(round(alpha_full, digits = 4))")
    println("  Subsampled: α=$(round(alpha_sub, digits = 4))")
    println("  Difference: $(round(abs(alpha_full - alpha_sub), digits = 4))")

    @test isfinite(alpha_full)
    @test isfinite(alpha_sub)
    # Results should be reasonably similar
    @test abs(alpha_full - alpha_sub) < 0.5
end

println("\n=== All Extended Subsampling Tests Complete ===")
println("Tested: Auto-detection on large datasets, batch size variations,")
println("        forced modes, accuracy comparisons with n>10000")
