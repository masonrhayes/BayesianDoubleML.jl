# AD Backends Smoke Test (CI)
# Minimal test to verify AD backends work - runs fast in CI
# Comprehensive tests are in test/extended/

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "AD Backend Smoke Test - AutoReverseDiff" begin
    Random.seed!(500)
    Y, D, X, alpha_true, _ = generate_dgp_table1(300, 5, 2.0; alpha_true = 0.5, rng = MersenneTwister(500))

    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)
    result = fit(problem, method; n_iterations = 100, n_draws = 100)

    alpha_mean = mean(extract_alpha(result))

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end

@testset "AD Backend Smoke Test - Basic Model" begin
    Random.seed!(501)
    Y, D, X, alpha_true, _ = generate_dgp_table1(300, 5, 2.0; alpha_true = 0.5, rng = MersenneTwister(501))

    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI(; ad_backend = AutoReverseDiff)
    result = fit(problem, method; n_iterations = 100, n_draws = 100)

    alpha_mean = mean(extract_alpha(result))

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - alpha_true) < 0.5
end
