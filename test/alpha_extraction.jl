# Alpha Extraction Tests
# Tests for all alpha extraction methods from MCMC and VI results via Model API
# Correct parameter indexing and extraction validation

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "Alpha Extraction from MCMC - Hierarchical Model" begin
    Random.seed!(800)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(800))
    alpha_true = 0.5

    println("\n=== Alpha Extraction: MCMC Hierarchical ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    # Check alpha_samples field exists on stored result
    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 200

    # Check all samples are finite
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  All samples finite: $(all(isfinite.(model.result.alpha_samples)))")
    println("  ✓ Alpha extraction from MCMC hierarchical works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from MCMC - Basic Model" begin
    Random.seed!(801)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.8, rng = MersenneTwister(801))
    alpha_true = 0.8

    println("\n=== Alpha Extraction: MCMC Basic ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 200
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from MCMC basic works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from VI - Hierarchical Model" begin
    Random.seed!(802)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(802))
    alpha_true = 0.5

    println("\n=== Alpha Extraction: VI Hierarchical ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()
    fit!(model, method; n_iterations = 500, n_draws = 1_000)

    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 500
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI hierarchical works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from VI - Basic Model" begin
    Random.seed!(803)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.6, rng = MersenneTwister(803))
    alpha_true = 0.6

    println("\n=== Alpha Extraction: VI Basic ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = UnifiedVI()
    fit!(model, method; n_iterations = 500, n_draws = 1_000)

    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 500
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI basic works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Extraction from SimpleVI" begin
    Random.seed!(804)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(804))
    alpha_true = 0.5

    println("\n=== Alpha Extraction: SimpleVI ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI()
    fit!(model, method; n_iterations = 500, n_draws = 1_000)

    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 500
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Extracted α mean: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from SimpleVI works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha Range Validation via Models" begin
    Random.seed!(805)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(805))
    alpha_true = 0.5

    println("\n=== Alpha Range Validation ===")

    # Test MCMC
    model_mcmc = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_mcmc, MCMCMethod(:nuts); n_samples = 200, n_chains = 2)

    @test minimum(model_mcmc.result.alpha_samples) > -10
    @test maximum(model_mcmc.result.alpha_samples) < 10

    # Test VI
    model_vi = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_vi, UnifiedVI())

    @test minimum(model_vi.result.alpha_samples) > -10
    @test maximum(model_vi.result.alpha_samples) < 10

    println("  MCMC α range: [$(round(minimum(model_mcmc.result.alpha_samples), digits = 4)), $(round(maximum(model_mcmc.result.alpha_samples), digits = 4))]")
    println("  VI α range: [$(round(minimum(model_vi.result.alpha_samples), digits = 4)), $(round(maximum(model_vi.result.alpha_samples), digits = 4))]")
    println("  ✓ Alpha values in reasonable range")
end

@testset "Alpha Statistics Consistency via Model" begin
    Random.seed!(806)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(806))
    alpha_true = 0.5

    println("\n=== Alpha Statistics Consistency ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200, n_chains = 3)

    # Manual calculation should match result.mean_alpha if available
    alpha_manual_mean = mean(extract_alpha(model))
    alpha_manual_std = std(extract_alpha(model))

    # Basic sanity checks
    @test alpha_manual_mean > -5
    @test alpha_manual_mean < 5
    @test alpha_manual_std > 0

    # CI should contain mean
    ci = credible_interval(model)
    @test ci[1] < alpha_manual_mean < ci[2]

    println("  Mean: $(round(alpha_manual_mean, digits = 4))")
    println("  Std: $(round(alpha_manual_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ✓ Alpha statistics consistent")
end

@testset "Alpha Extraction with Binary Treatment via Model" begin
    Random.seed!(807)
    Y, D, X, alpha_true = make_binary_treatment_data(n = 100, p = 10, alpha_true = 0.5, seed = 807)

    println("\n=== Alpha Extraction: Binary Treatment ===")
    println("True α: $alpha_true")

    @test all(D .∈ Ref([0.0, 1.0]))

    # MCMC
    model_mcmc = BDMLModel(Y, D, X; model_type = :hier)
    fit!(model_mcmc, MCMCMethod(:nuts); n_samples = 200, n_chains = 2)

    alpha_mcmc = mean(model_mcmc.result.alpha_samples)

    # VI
    model_vi = BDMLModel(Y, D, X; model_type = :hier)
    fit!(model_vi, UnifiedVI())

    alpha_vi = mean(model_vi.result.alpha_samples)

    println("  MCMC α: $(round(alpha_mcmc, digits = 4))")
    println("  VI α: $(round(alpha_vi, digits = 4))")
    println("  ✓ Alpha extraction with binary treatment works")

    @test isfinite(alpha_mcmc)
    @test isfinite(alpha_vi)
end

@testset "Alpha Extraction with Different Seeds via Model" begin
    Random.seed!(808)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(808))
    alpha_true = 0.5
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])

    println("\n=== Alpha Extraction: Different Seeds ===")

    alphas = Float64[]

    for seed in [100, 200, 300]
        Random.seed!(seed)
        model = BDMLModel(df, :y, :d; model_type = :hier)
        method = UnifiedVI()
        fit!(model, method; n_iterations = 500, n_draws = 1_000, force = true)

        alpha_mean = mean(extract_alpha(model))
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

@testset "Alpha Extraction with Multi-Chain MCMC via Model" begin
    Random.seed!(809)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(809))
    alpha_true = 0.5

    println("\n=== Alpha Extraction: Multi-Chain MCMC ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 150, n_chains = 2)

    @test length(model.result.alpha_samples) >= 300  # 150 * 2
    @test all(isfinite.(model.result.alpha_samples))

    alpha_mean = mean(extract_alpha(model))

    println("  Chains: 2")
    println("  Total samples: $(length(model.result.alpha_samples))")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from multi-chain works")

    @test isfinite(alpha_mean)
end

@testset "Alpha Posterior Distribution Shape via Model" begin
    Random.seed!(810)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(810))
    alpha_true = 0.5

    println("\n=== Alpha Posterior Distribution Shape ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    samples = model.result.alpha_samples

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

@testset "Alpha Coefficient Table Extraction via Model" begin
    Random.seed!(811)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(811))
    alpha_true = 0.5

    println("\n=== Alpha from Coefficient Table ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    ct = coeftable(model)

    # Find alpha in coefficient table
    alpha_idx = findfirst(contains("α"), ct.coefnames)
    @test alpha_idx !== nothing

    alpha_from_ct = ct.coef[alpha_idx]
    alpha_from_samples = mean(extract_alpha(model))

    println("  From samples: $(round(alpha_from_samples, digits = 4))")
    println("  From coeftable: $(round(alpha_from_ct, digits = 4))")
    println("  ✓ Alpha accessible from coefficient table")

    # These should match closely
    @test abs(alpha_from_ct - alpha_from_samples) < 0.01
end

@testset "Alpha from MCMC Result via Model" begin
    Random.seed!(812)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(812))
    alpha_true = 0.5

    println("\n=== Alpha from MCMC Result ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, MCMCNUTS(); n_samples = 200)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from MCMC works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "Alpha from VI Result via Model" begin
    Random.seed!(813)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(813))
    alpha_true = 0.5

    println("\n=== Alpha from VI Result ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model, UnifiedVIMethod(); n_iterations = 500, n_draws = 1_000, show_progress = false)

    @test hasfield(typeof(model.result), :alpha_samples)
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ✓ Alpha extraction from VI works")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

println("\n=== All Alpha Extraction Tests Complete ===")
println("Tested: MCMC, VI, SimpleVI, binary treatment, multi-chain via Model API")
