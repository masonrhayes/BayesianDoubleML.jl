# Subsampling Tests (CI-Friendly)
# Tests for subsampling functionality using small datasets with forced subsampling
# Large dataset tests (n>10000) are in test/extended/subsampling_extended.jl

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "Small Dataset - Force Subsampling on Small Data" begin
    Random.seed!(601)
    n, p = 100, 5
    df = make_plr_DTL2025(n, p, 2.0; alpha = 0.5, rng = MersenneTwister(601))
    alpha_true = 0.5

    println("\n=== Small Dataset: Force Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI(;
        subsample = true,
        batch_size = 32
    )

    fit!(model, method; n_iterations = 100, n_draws = 100)

    alpha_mean = mean(extract_alpha(model))

    println("  Batch size: 32")
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Small Dataset - No Subsampling (Auto)" begin
    Random.seed!(600)
    n, p = 100, 5
    df = make_plr_DTL2025(n, p, 2.0; alpha = 0.5, rng = MersenneTwister(600))
    alpha_true = 0.5

    println("\n=== Small Dataset: Auto No Subsampling (n=$n) ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()

    fit!(model, method; n_iterations = 100, n_draws = 100)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfitted(model)
    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "BDMLVIModel Subsampling Interface" begin
    Random.seed!(606)
    n, p = 100, 5
    df = make_plr_DTL2025(n, p, 2.0; alpha = 0.5, rng = MersenneTwister(606))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])
    alpha_true = 0.5

    println("\n=== BDMLVIModel Subsampling Interface ===")

    # Create BDMLVIModel
    vi_model = BayesianDoubleML.BDMLVIModel(Y, D, X; model_type = :hier)

    @test vi_model.n_data == n
    @test vi_model.model_type == :hier

    # Test subsampling
    idx = 1:50
    using AdvancedVI
    model_sub = AdvancedVI.subsample(vi_model, idx)

    @test model_sub.n_data == n  # Original preserved
    @test length(model_sub.Y) == 50
    @test size(model_sub.X) == (50, p)

    println("  Original n: $n")
    println("  Subsampled n: $(length(model_sub.Y))")
    println("  ✓ Subsampling interface works correctly")
end

@testset "Small Dataset with Basic Model - Force Subsampling" begin
    Random.seed!(608)
    n, p = 100, 5
    df = make_plr_DTL2025(n, p, 2.0; alpha = 0.5, rng = MersenneTwister(608))
    alpha_true = 0.5

    println("\n=== Small Dataset with Basic Model: Force Subsampling ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = UnifiedVI(;
        subsample = true,
        batch_size = 32
    )
    fit!(model, method; n_iterations = 100, n_draws = 100)
    alpha_mean = mean(extract_alpha(model))

    println("  Batch size: 32")
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfitted(model)
    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

println("\n=== All Subsampling Tests Complete ===")
println("Tested: Forced subsampling on small datasets, interface validation with new Model API")
println("Note: Large dataset tests (n>10000) are in test/extended/subsampling_extended.jl")
