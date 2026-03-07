# Subsampling Tests
# Tests for automatic and manual subsampling in VI
# Batch size computation, thresholds, and large dataset handling

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "Subsampling Threshold Detection" begin
    println("\n=== Subsampling Threshold Detection ===")

    # Test the should_use_subsampling function
    @test BayesianDoubleML.should_use_subsampling(500) == false
    @test BayesianDoubleML.should_use_subsampling(1000) == false
    @test BayesianDoubleML.should_use_subsampling(5000) == false
    @test BayesianDoubleML.should_use_subsampling(9999) == false
    @test BayesianDoubleML.should_use_subsampling(10000) == true
    @test BayesianDoubleML.should_use_subsampling(50000) == true
    @test BayesianDoubleML.should_use_subsampling(100000) == true

    println("  n < 10000: no subsampling")
    println("  n >= 10000: use subsampling")
    println("  ✓ Threshold detection works correctly")
end

@testset "Batch Size Computation" begin
    println("\n=== Batch Size Computation ===")

    # Test compute_batch_size function
    @test BayesianDoubleML.compute_batch_size(500) == 64      # Minimum
    @test BayesianDoubleML.compute_batch_size(1000) == 64     # Minimum
    @test BayesianDoubleML.compute_batch_size(6400) == 64     # n/100 = 64
    @test BayesianDoubleML.compute_batch_size(10000) == 64    # n/100 = 100, but min is 64
    @test BayesianDoubleML.compute_batch_size(20000) == 128   # Capped logic
    @test BayesianDoubleML.compute_batch_size(50000) == 256   # Maximum cap
    @test BayesianDoubleML.compute_batch_size(100000) == 256  # Maximum cap

    println("  Small n (500): batch = 64 (minimum)")
    println("  Medium n (10000): batch = 64-128")
    println("  Large n (50000+): batch = 256 (maximum)")
    println("  ✓ Batch size computation works correctly")
end

@testset "Small Dataset - No Subsampling (Auto)" begin
    Random.seed!(600)
    n, p = 500, 10
    Y, D, X, alpha_true = make_test_data(n = n, p = p, alpha_true = 0.5, seed = 600)

    println("\n=== Small Dataset: Auto No Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    @test BayesianDoubleML.should_use_subsampling(n) == false

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Small Dataset - Force Subsampling" begin
    Random.seed!(601)
    n, p = 500, 10
    Y, D, X, alpha_true = make_test_data(n = n, p = p, alpha_true = 0.5, seed = 601)

    println("\n=== Small Dataset: Force Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(;
        subsample = true,
        batch_size = 64,
        n_iterations = 500,
        n_draws = 500
    )

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Medium Dataset - Auto Subsampling Triggered" begin
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

    @test BayesianDoubleML.should_use_subsampling(n) == true

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

@testset "Large Dataset - Explicit Subsampling Control" begin
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
        batch_size = 128,
        n_iterations = 400,
        n_draws = 400
    )

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Batch size: 128")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Subsampling Batch Size Variations" begin
    Random.seed!(604)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling: Batch Size Variations ===")
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

@testset "Force Full-Batch (subsample=false)" begin
    Random.seed!(605)
    n, p = 12000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Force Full-Batch (subsample=false) ===")
    println("True α: $alpha_true")
    println("Dataset: n=$n (would auto-subsample)")

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(;
        subsample = false,  # Force full-batch
        n_iterations = 200,  # Reduced for speed
        n_draws = 200
    )

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(result))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "BDMLVIModel Subsampling Interface" begin
    Random.seed!(606)
    n, p = 1000, 10
    Y, D, X, alpha_true = make_test_data(n = n, p = p, alpha_true = 0.5, seed = 606)

    println("\n=== BDMLVIModel Subsampling Interface ===")

    # Create BDMLVIModel
    model = BayesianDoubleML.BDMLVIModel(Y, D, X; model_type = :hier)

    @test model.n_data == n
    @test model.model_type == :hier

    # Test subsampling
    idx = 1:100
    using AdvancedVI
    model_sub = AdvancedVI.subsample(model, idx)

    @test model_sub.n_data == n  # Original preserved
    @test length(model_sub.Y) == 100
    @test size(model_sub.X) == (100, p)

    println("  Original n: $n")
    println("  Subsampled n: $(length(model_sub.Y))")
    println("  ✓ Subsampling interface works correctly")
end

@testset "Subsampling with Different AD Backends" begin
    Random.seed!(607)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling with Different AD Backends ===")
    println("True α: $alpha_true")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    backends = [AutoReverseDiff, AutoMooncake]

    for backend in backends
        Random.seed!(607)

        # Warmup for Mooncake
        if backend == AutoMooncake
            _ = fit(problem, UnifiedVI(; ad_backend = AutoMooncake, n_iterations = 50, n_draws = 100))
            Random.seed!(607)
        end

        method = UnifiedVI(;
            ad_backend = backend,
            subsample = true,
            batch_size = 128,
            n_iterations = 300,
            n_draws = 300
        )

        elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)
        alpha_mean = mean(extract_alpha(result))

        println("  $(nameof(backend)): α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")

        @test isfinite(alpha_mean)
        @test abs(alpha_mean - alpha_true) < 0.5
    end
end

@testset "Subsampling with Basic Model" begin
    Random.seed!(608)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling with Basic Model ===")
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

@testset "Subsampling Accuracy Comparison" begin
    Random.seed!(609)
    n, p = 10000, 15
    X = randn(n, p)
    gamma = randn(p) .* 0.3
    D = X * gamma + randn(n) .* 0.5
    beta = randn(p) .* 0.2
    Y = 0.5 .* D + X * beta + randn(n)
    alpha_true = 0.5

    println("\n=== Subsampling Accuracy Comparison ===")
    println("True α: $alpha_true")
    println("Dataset: n=$n")

    problem = BDMLProblem(Y, D, X; model_type = :hier)

    # Full batch (forced)
    Random.seed!(609)
    method_full = UnifiedVI(; subsample = false, n_iterations = 200, n_draws = 200)
    result_full = fit(problem, method_full)
    alpha_full = mean(result_full.alpha_samples)

    # Subsampled
    Random.seed!(609)
    method_sub = UnifiedVI(; subsample = true, batch_size = 128)
    result_sub = fit(problem, method_sub)
    alpha_sub = mean(result_sub.alpha_samples)

    println("  Full batch: α=$(round(alpha_full, digits = 4))")
    println("  Subsampled: α=$(round(alpha_sub, digits = 4))")
    println("  Difference: $(round(abs(alpha_full - alpha_sub), digits = 4))")

    @test isfinite(alpha_full)
    @test isfinite(alpha_sub)
    # Results should be reasonably similar
    @test abs(alpha_full - alpha_sub) < 0.5
end

println("\n=== All Subsampling Tests Complete ===")
println("Tested: Auto-detection, batch sizes, thresholds, forced modes, accuracy")
