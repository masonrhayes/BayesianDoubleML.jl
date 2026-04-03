# Core Types and Utilities Tests
# Tests for BDMLData, standardization, and basic utility functions

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "BDMLData Construction" begin
    df = make_plr_DTL2025(50, 5, 2.0; alpha = 0.5, rng = MersenneTwister(100))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])

    data = BDMLData(Y, D, X)

    @test data.n == 50
    @test data.p == 5
    @test length(data.Y) == 50
    @test length(data.D) == 50
    @test size(data.X) == (50, 5)

    # Test that data is stored correctly
    @test data.Y ≈ Y
    @test data.D ≈ D
    @test data.X ≈ X
end

@testset "BDMLData Type Conversion" begin
    # Test with integer data (should convert to Float64)
    Y_int = collect(Int, 1:50)
    D_int = collect(Int, 51:100)
    X_int = reshape(collect(Int, 1:250), 50, 5)

    data = BDMLData(Y_int, D_int, X_int)

    @test eltype(data.Y) == Float64
    @test eltype(data.D) == Float64
    @test eltype(data.X) == Float64
    @test data.Y ≈ Float64.(Y_int)
end

@testset "BDMLData Error Handling" begin
    df = make_plr_DTL2025(50, 5, 2.0; alpha = 0.5, rng = MersenneTwister(101))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])

    # Dimension mismatch between Y and D
    @test_throws AssertionError BDMLData(Y[1:25], D, X)

    # Dimension mismatch between D and X rows
    @test_throws AssertionError BDMLData(Y, D[1:25], X)

    # Dimension mismatch between X rows and Y
    X_wrong = randn(25, 5)
    @test_throws AssertionError BDMLData(Y, D, X_wrong)
end

@testset "Standardization" begin
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.5, rng = MersenneTwister(102))
    Y = df.y
    D = df.d
    X = Matrix(df[:, r"^X"])

    Y_s, D_s, X_s, stats = BayesianDoubleML.standardize_data(Y, D, X)

    # Check means are zero (within numerical precision)
    @test abs(mean(Y_s)) < 1.0e-10
    @test abs(mean(D_s)) < 1.0e-10
    @test all(abs.(mean(X_s, dims = 1)) .< 1.0e-10)

    # Check standard deviations are 1
    @test abs(std(Y_s) - 1.0) < 1.0e-10
    @test abs(std(D_s) - 1.0) < 1.0e-10

    # Check stats structure
    @test stats.Y_mean ≈ mean(Y)
    @test stats.Y_sd ≈ std(Y)
    @test stats.D_mean ≈ mean(D)
    @test stats.D_sd ≈ std(D)
    @test stats.X_mean ≈ vec(mean(X, dims = 1))
    @test stats.X_sd ≈ vec(std(X, dims = 1))
end

@testset "Credible Interval" begin
    # Test with normal samples
    Random.seed!(103)
    samples = randn(10000) .+ 0.5  # Mean 0.5, std 1.0

    ci = BayesianDoubleML.credible_interval(samples; level = 0.95)
    @test length(ci) == 2
    @test ci[1] < ci[2]

    # 95% CI for N(0.5, 1) should be roughly [-1.46, 2.46]
    @test ci[1] ≈ -1.46 atol = 0.1
    @test ci[2] ≈ 2.46 atol = 0.1

    # CI should contain the mean
    @test ci[1] < mean(samples) < ci[2]

    # 90% CI should be narrower than 95% CI
    ci_90 = BayesianDoubleML.credible_interval(samples; level = 0.9)
    @test ci_90[1] > ci[1]  # Lower bound higher
    @test ci_90[2] < ci[2]  # Upper bound lower
end

@testset "Convergence Checking" begin
    # Converging sequence (monotonic increasing with decreasing increments)
    # Use larger sequence to ensure window >= 10 iterations (function uses max(10, ...))
    elbo_conv = Float64[100, 110, 115, 118, 119.5, 119.8, 119.9, 119.95, 119.98, 119.99, 119.995, 119.998, 119.999, 119.9995, 119.9998]
    conv, msg = BayesianDoubleML.check_elbo_convergence(
        elbo_conv;
        min_pct = 0.3,
        rel_tol = 0.01,  # 1% relative tolerance
        check_trend = true,
        min_iterations = 10
    )
    @test conv == true

    # Too short (less than min_iterations)
    elbo_short = Float64[100, 105]
    conv_short, _ = BayesianDoubleML.check_elbo_convergence(
        elbo_short;
        min_pct = 0.5,
        rel_tol = 1.0e-3,
        check_trend = false,
        min_iterations = 5
    )
    @test conv_short == false

    # Oscillating sequence (not converged)
    elbo_osc = Float64[100, 110, 105, 115, 110, 120, 115, 108, 118, 112, 122, 116]
    conv_osc, _ = BayesianDoubleML.check_elbo_convergence(
        elbo_osc;
        min_pct = 0.3,
        rel_tol = 0.001,
        check_trend = true,
        min_iterations = 10
    )
    @test conv_osc == false

    # Plateau (converged) - use larger sequence for window calculation
    elbo_plateau = Float64[100, 100.01, 100.02, 100.015, 100.018, 100.017, 100.019, 100.018, 100.02, 100.019, 100.021, 100.02]
    conv_plateau, _ = BayesianDoubleML.check_elbo_convergence(
        elbo_plateau;
        min_pct = 0.3,
        rel_tol = 0.001,
        check_trend = true,
        min_iterations = 10
    )
    @test conv_plateau == true
end

@testset "Posterior Summary" begin
    Random.seed!(104)
    samples = randn(1000) .+ 2.0

    summary = BayesianDoubleML.posterior_summary(samples)

    @test summary[:mean] ≈ mean(samples)
    @test summary[:std] ≈ std(samples)
    @test length(summary[:quantiles]) == 5
    @test summary[:prob_range] == [0.025, 0.25, 0.5, 0.75, 0.975]
end
