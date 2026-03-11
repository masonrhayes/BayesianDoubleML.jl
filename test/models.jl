# BDMLModel Constructors and Accessors Tests
# Tests for BDMLModel, BDMLBasicModel, BDMLHierarchicalModel

using BayesianDoubleML
using Test
using Random
using Statistics

# Load test utilities
include("utils.jl")

@testset "BDMLModel Basic Constructor" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(200))

    model = BDMLModel(Y, D, X; model_type = :basic)

    @test model isa BDMLBasicModel
    @test nobs(model) == 100
    @test ncovariates(model) == 10
    @test model_type(model) == :basic
    @test !isfitted(model)

    # Data should be standardized in-place
    @test abs(mean(model.Y)) < 1.0e-10
    @test abs(mean(model.D)) < 1.0e-10

    # Stats should be preserved
    stats = BayesianDoubleML.standardization_stats(model)
    @test stats isa BayesianDoubleML.StandardizationStats
    @test stats.Y_mean ≈ mean(Y)
    @test stats.Y_sd ≈ std(Y)
end

@testset "BDMLModel Hierarchical Constructor" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(201))

    model = BDMLModel(Y, D, X; model_type = :hier)

    @test model isa BDMLHierarchicalModel
    @test model_type(model) == :hier
    @test nobs(model) == 100
    @test ncovariates(model) == 10
    @test !isfitted(model)

    # Data should be standardized
    @test abs(mean(model.Y)) < 1.0e-10
    @test abs(mean(model.D)) < 1.0e-10
end

@testset "BDMLModel from BDMLData" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(202))
    data = BDMLData(Y, D, X)

    model = BDMLModel(data; model_type = :basic)
    @test model isa BDMLBasicModel
    @test nobs(model) == 100
    @test ncovariates(model) == 10
    @test !isfitted(model)
end

@testset "BDMLModel Error Handling" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(203))

    # Invalid model_type
    @test_throws ArgumentError BDMLModel(Y, D, X; model_type = :invalid)
    @test_throws ArgumentError BDMLModel(Y, D, X; model_type = :hierarchical)  # Not :hier

    # Dimension mismatch between Y and D
    @test_throws AssertionError BDMLModel(Y[1:50], D, X)

    # Dimension mismatch between D and X rows
    @test_throws AssertionError BDMLModel(Y, D[1:50], X)
end

@testset "Pre-allocated Temporaries" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(204))

    # Basic model
    model_basic = BDMLModel(Y, D, X; model_type = :basic)
    @test length(model_basic.μ_Y_cache) == 100
    @test length(model_basic.μ_D_cache) == 100

    # Hierarchical model
    model_hier = BDMLModel(Y, D, X; model_type = :hier)
    @test length(model_hier.μ_Y_cache) == 100
    @test length(model_hier.μ_D_cache) == 100
end

@testset "Model Data Handling" begin
    Y, D, X, _ = generate_dgp_table1(100, 10, 2.0; alpha_true = 0.5, rng = MersenneTwister(205))
    Y_original = copy(Y)
    D_original = copy(D)

    model = BDMLModel(Y, D, X; model_type = :basic)

    # Original data should NOT be modified (standardization creates copies)
    @test Y ≈ Y_original
    @test D ≈ D_original

    # But model data should be standardized
    @test abs(mean(model.Y)) < 1.0e-10
    @test abs(mean(model.D)) < 1.0e-10
end

@testset "Small Dataset Model" begin
    Y, D, X, _ = generate_dgp_table1(20, 3, 2.0; alpha_true = 0.5, rng = MersenneTwister(206))

    model = BDMLModel(Y, D, X; model_type = :basic)

    @test nobs(model) == 20
    @test ncovariates(model) == 3
    @test !isfitted(model)
end

@testset "Large Dataset Model" begin
    Y, D, X, _ = generate_dgp_table1(10000, 50, 2.0; alpha_true = 0.5, rng = MersenneTwister(207))

    model = BDMLModel(Y, D, X; model_type = :hier)

    @test nobs(model) == 10000
    @test ncovariates(model) == 50
    @test !isfitted(model)
end

@testset "Model Fitted State" begin
    Y, D, X, _ = generate_dgp_table1(50, 5, 2.0; alpha_true = 0.5, rng = MersenneTwister(208))

    model = BDMLModel(Y, D, X; model_type = :hier)

    # Before fitting
    @test !isfitted(model)
    @test model.result === nothing
    @test model.last_method === nothing
    @test model.is_fitted == false

    # After fitting (using default MCMC with few samples for speed)
    fit!(model; n_samples = 100, n_chains = 1)

    # After fitting
    @test isfitted(model)
    @test model.is_fitted == true
    @test model.result !== nothing
    @test model.last_method !== nothing
    @test model.result isa BDMLMCMCResult
end

@testset "Model Refitting with force" begin
    Y, D, X, _ = generate_dgp_table1(50, 5, 2.0; alpha_true = 0.5, rng = MersenneTwister(209))

    model = BDMLModel(Y, D, X; model_type = :basic)

    # First fit
    fit!(model; n_samples = 100, n_chains = 1)
    @test isfitted(model)
    first_result = model.result

    # Try to fit again without force - should warn and do nothing
    @test_logs (:warn, "Model has already been fitted. Use force=true to refit.") fit!(model; n_samples = 100, n_chains = 1)
    @test model.result === first_result  # Same result object

    # Fit again with force - should refit
    fit!(model; n_samples = 100, n_chains = 1, force = true)
    @test isfitted(model)
    @test model.result !== first_result  # Different result object
end

@testset "Model Extraction Functions" begin
    Y, D, X, _ = generate_dgp_table1(50, 5, 2.0; alpha_true = 0.5, rng = MersenneTwister(210))

    model = BDMLModel(Y, D, X; model_type = :hier)

    # Before fitting - extraction should error
    @test_throws ErrorException coeftable(model)
    @test_throws ErrorException extract_alpha(model)
    @test_throws ErrorException coef(model)
    @test_throws ErrorException stderror(model)

    # After fitting - extraction should work
    fit!(model; n_samples = 100, n_chains = 1)

    @test_nowarn coeftable(model)
    @test_nowarn extract_alpha(model)
    @test_nowarn coef(model)
    @test_nowarn stderror(model)
    @test_nowarn confint(model)
    @test_nowarn pvalues(model)
end
