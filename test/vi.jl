# VI Tests (Consolidated)
# Tests for both UnifiedVI and SimpleVI inference methods with fit!()
# Basic and hierarchical models with various configurations

using BayesianDoubleML
using Mooncake
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

# ============================================================================
# UnifiedVI Tests
# ============================================================================

@testset "UnifiedVI Basic Model" begin
    Random.seed!(300)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(300))
    alpha_true = 0.5

    println("\n=== UnifiedVI: Basic Model ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = UnifiedVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :basic
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))
    ci = credible_interval(model)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Converged: $(model.result.converged)")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
    @test isfinite(model.result.final_elbo)
end

@testset "UnifiedVI Hierarchical Model" begin
    Random.seed!(301)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.8, rng = MersenneTwister(301))
    alpha_true = 0.8

    println("\n=== UnifiedVI: Hierarchical Model ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :hier
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))
    ci = credible_interval(model)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI with BDMLData" begin
    Random.seed!(302)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(302))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])
    alpha_true = 0.5

    println("\n=== UnifiedVI with BDMLData ===")

    data = BDMLData(Y, D, X)
    model = BDMLModel(data; model_type = :hier)
    method = UnifiedVI()

    fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "UnifiedVI AD Backend - AutoReverseDiff" begin
    Random.seed!(303)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(303))
    alpha_true = 0.5

    println("\n=== UnifiedVI: AutoReverseDiff ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)

    elapsed = @elapsed fit!(model, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI AD Backend - AutoForwardDiff" begin
    Random.seed!(304)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(304))
    alpha_true = 0.5

    println("\n=== UnifiedVI: AutoForwardDiff ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoForwardDiff)

    elapsed = @elapsed fit!(model, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI AD Backend - AutoZygote" begin
    import Zygote

    Random.seed!(306)
    df = make_plr_DTL2025(80, 8, 2.0; alpha = 0.5, rng = MersenneTwister(306))
    alpha_true = 0.5

    println("\n=== UnifiedVI: AutoZygote ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoZygote)

    elapsed = @elapsed fit!(model, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "UnifiedVI with Subsampling Enabled" begin
    Random.seed!(307)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(307))
    alpha_true = 0.5

    println("\n=== UnifiedVI with Subsampling ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI(; subsample = true, batch_size = 64)

    fit!(model, method; n_iterations = 400, n_draws = 400)

    @test isfitted(model)
    @test model.result isa BDMLVIResult

    alpha_mean = mean(extract_alpha(model))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "UnifiedVI Monte Carlo Samples Variation" begin
    Random.seed!(308)
    df = make_plr_DTL2025(80, 8, 2.0; alpha = 0.5, rng = MersenneTwister(308))
    alpha_true = 0.5

    println("\n=== UnifiedVI: n_montecarlo Variation ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)

    for n_mc in [5, 10, 20]
        method = UnifiedVI(; n_montecarlo = n_mc)
        fit!(model, method; n_iterations = 300, n_draws = 300, force = true)

        alpha_mean = mean(extract_alpha(model))
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

    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    fit!(model, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(model))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "UnifiedVI Result Display and Credible Interval" begin
    Random.seed!(310)
    df = make_plr_DTL2025(50, 5, 2.0; alpha = 0.5, rng = MersenneTwister(310))
    alpha_true = 0.5

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()

    fit!(model, method; n_iterations = 300, n_draws = 300)

    # Test display methods on model
    @test_nowarn println(model)
    @test_nowarn show(model)

    # Test display methods on result
    @test_nowarn println(model.result)
    @test_nowarn show(model.result)

    # Test credible interval on model
    ci = credible_interval(model)
    @test length(ci) == 2
    @test ci[1] < ci[2]

    alpha_mean = mean(extract_alpha(model))
    @test ci[1] < alpha_mean < ci[2]
end

@testset "UnifiedVI Convergence Detection" begin
    Random.seed!(311)
    df = make_plr_DTL2025(80, 8, 2.0; alpha = 0.5, rng = MersenneTwister(311))
    alpha_true = 0.5

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = UnifiedVI()

    fit!(model, method; n_iterations = 500, n_draws = 500)

    @test hasfield(typeof(model.result), :converged)
    @test hasfield(typeof(model.result), :final_elbo)
    @test isfinite(model.result.final_elbo)

    println("  Converged: $(model.result.converged)")
    println("  Final ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Iterations: $(model.result.n_iterations)")
end

# ============================================================================
# SimpleVI Tests
# ============================================================================

@testset "SimpleVI Basic Model" begin
    Random.seed!(400)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(400))
    alpha_true = 0.5

    println("\n=== SimpleVI: Basic Model ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = SimpleVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :basic
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))
    ci = credible_interval(model)

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
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.8, rng = MersenneTwister(401))
    alpha_true = 0.8

    println("\n=== SimpleVI: Hierarchical Model ===")
    println("True α: $alpha_true")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :hier
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI with BDMLData" begin
    Random.seed!(402)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(402))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])
    alpha_true = 0.5

    println("\n=== SimpleVI with BDMLData ===")

    data = BDMLData(Y, D, X)
    model = BDMLModel(data; model_type = :hier)
    method = SimpleVI()

    fit!(model, method; n_iterations = 500, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
end

@testset "SimpleVI AD Backend - AutoMooncake" begin
    Random.seed!(403)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(403))
    alpha_true = 0.5

    println("\n=== SimpleVI: AutoMooncake Backend ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI(; ad_backend = AutoMooncake)

    # Warmup
    println("  Warmup...")
    fit!(model, SimpleVI(; ad_backend = AutoMooncake); n_iterations = 50, n_draws = 100, force = true)

    elapsed = @elapsed fit!(model, method; n_iterations = 400, n_draws = 400, force = true)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI AD Backend - AutoReverseDiff" begin
    Random.seed!(404)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(404))
    alpha_true = 0.5

    println("\n=== SimpleVI: AutoReverseDiff Backend ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI(; ad_backend = AutoReverseDiff)

    elapsed = @elapsed fit!(model, method; n_iterations = 400, n_draws = 400)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI AD Backend - AutoZygote" begin
    using Zygote

    Random.seed!(410)
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(410))
    alpha_true = 0.5

    println("\n=== SimpleVI: AutoZygote Backend ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI(; ad_backend = AutoZygote)

    elapsed = @elapsed fit!(model, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(extract_alpha(model))

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

    model = BDMLModel(Y, D, X; model_type = :hier)
    method = SimpleVI()

    fit!(model, method; n_iterations = 500, n_draws = 500)

    alpha_mean = mean(extract_alpha(model))
    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "SimpleVI Small Dataset" begin
    Random.seed!(406)
    df = make_plr_DTL2025(50, 5, 2.0; alpha = 0.5, rng = MersenneTwister(406))
    alpha_true = 0.5

    println("\n=== SimpleVI: Small Dataset (n=50, p=5) ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI()

    fit!(model, method; n_iterations = 300, n_draws = 300)

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.6
end

@testset "SimpleVI Result Display" begin
    Random.seed!(407)
    df = make_plr_DTL2025(50, 5, 2.0; alpha = 0.5, rng = MersenneTwister(407))
    alpha_true = 0.5

    model = BDMLModel(df, :y, :d; model_type = :hier)
    method = SimpleVI()

    fit!(model, method; n_iterations = 300, n_draws = 300)

    # Test display methods on model
    @test_nowarn println(model)
    @test_nowarn show(model)

    # Test display methods on result
    @test_nowarn println(model.result)
    @test_nowarn show(model.result)

    ci = credible_interval(model)
    @test length(ci) == 2

    alpha_mean = mean(extract_alpha(model))
    @test ci[1] < alpha_mean < ci[2]
end

@testset "SimpleVI Different Iterations" begin
    Random.seed!(409)
    df = make_plr_DTL2025(80, 8, 2.0; alpha = 0.5, rng = MersenneTwister(409))
    alpha_true = 0.5

    println("\n=== SimpleVI: Different Iteration Counts ===")

    model = BDMLModel(df, :y, :d; model_type = :hier)

    for n_iter in [200, 400]
        method = SimpleVI()
        fit!(model, method; n_iterations = n_iter, n_draws = 400, force = true)

        alpha_mean = mean(extract_alpha(model))
        println("  n_iterations=$n_iter: α = $(round(alpha_mean, digits = 4))")

        @test isfinite(alpha_mean)
    end
end
