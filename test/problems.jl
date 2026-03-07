# BDMLProblem Constructors and Accessors Tests
# Tests for BDMLProblem, BDMLBasicProblem, BDMLHierarchicalProblem

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "BDMLProblem Basic Constructor" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 200)

    prob = BDMLProblem(Y, D, X; model_type = :basic)

    @test prob isa BDMLBasicProblem
    @test nobs(prob) == 100
    @test ncovariates(prob) == 10
    @test model_type(prob) == :basic

    # Data should be standardized in-place
    @test abs(mean(prob.Y)) < 1.0e-10
    @test abs(mean(prob.D)) < 1.0e-10

    # Stats should be preserved
    stats = BayesianDoubleML.standardization_stats(prob)
    @test stats isa BayesianDoubleML.StandardizationStats
    @test stats.Y_mean ≈ mean(Y)
    @test stats.Y_sd ≈ std(Y)
end

@testset "BDMLProblem Hierarchical Constructor" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 201)

    prob = BDMLProblem(Y, D, X; model_type = :hier)

    @test prob isa BDMLHierarchicalProblem
    @test model_type(prob) == :hier
    @test nobs(prob) == 100
    @test ncovariates(prob) == 10

    # Data should be standardized
    @test abs(mean(prob.Y)) < 1.0e-10
    @test abs(mean(prob.D)) < 1.0e-10
end

@testset "BDMLProblem from BDMLData" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 202)
    data = BDMLData(Y, D, X)

    prob = BDMLProblem(data; model_type = :basic)
    @test prob isa BDMLBasicProblem
    @test nobs(prob) == 100
    @test ncovariates(prob) == 10
end

@testset "BDMLProblem Error Handling" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 203)

    # Invalid model_type
    @test_throws ArgumentError BDMLProblem(Y, D, X; model_type = :invalid)
    @test_throws ArgumentError BDMLProblem(Y, D, X; model_type = :hierarchical)  # Not :hier

    # Dimension mismatch between Y and D
    @test_throws AssertionError BDMLProblem(Y[1:50], D, X)

    # Dimension mismatch between D and X rows
    @test_throws AssertionError BDMLProblem(Y, D[1:50], X)
end

@testset "Pre-allocated Temporaries" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 204)

    # Basic problem
    prob_basic = BDMLProblem(Y, D, X; model_type = :basic)
    @test length(prob_basic.μ_Y_cache) == 100
    @test length(prob_basic.μ_D_cache) == 100

    # Hierarchical problem
    prob_hier = BDMLProblem(Y, D, X; model_type = :hier)
    @test length(prob_hier.μ_Y_cache) == 100
    @test length(prob_hier.μ_D_cache) == 100
end

@testset "Problem Data Handling" begin
    Y, D, X, _ = make_test_data(n = 100, p = 10, seed = 205)
    Y_original = copy(Y)
    D_original = copy(D)

    prob = BDMLProblem(Y, D, X; model_type = :basic)

    # Original data should NOT be modified (standardization creates copies)
    @test Y ≈ Y_original
    @test D ≈ D_original

    # But problem data should be standardized
    @test abs(mean(prob.Y)) < 1.0e-10
    @test abs(mean(prob.D)) < 1.0e-10
end

@testset "Small Dataset Problem" begin
    Y, D, X, _ = make_test_data(n = 20, p = 3, seed = 206)

    prob = BDMLProblem(Y, D, X; model_type = :basic)

    @test nobs(prob) == 20
    @test ncovariates(prob) == 3
end

@testset "Large Dataset Problem" begin
    Y, D, X, _ = make_test_data(n = 10000, p = 50, seed = 207)

    prob = BDMLProblem(Y, D, X; model_type = :hier)

    @test nobs(prob) == 10000
    @test ncovariates(prob) == 50
end
