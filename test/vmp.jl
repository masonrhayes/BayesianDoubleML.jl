# Variational Message Passing tests (RxInfer package extension)

using BayesianDoubleML
using RxInfer
using Random
using Statistics
using Test

@testset "VMP extension activation" begin
    @test Base.get_extension(BayesianDoubleML, :BayesianDoubleMLRxInferExt) !== nothing
end

@testset "VMP basic and hierarchical models" begin
    Random.seed!(8128)
    df = make_plr_DTL2025(100, 10, 1.0; alpha = 2.0)

    for model_type in (:basic, :hier)
        model = BDMLModel(df, :y, :d; model_type)
        fit!(model, VMP(); n_iterations = 50, n_draws = 300, seed = 42)

        @test isfitted(model)
        @test model.result isa BDMLVIResult
        @test model.result.vi_method == :vmp
        @test model.result.variational_family == :structured
        @test model.result.n_iterations == 50
        @test length(model.result.alpha_samples) == 300
        @test all(isfinite, model.result.alpha_samples)
        @test abs(mean(model.result.alpha_samples) - 2.0) < 0.5
        @test model.result.converged

        # Bethe Free Energy is stored as ELBO = -BFE and should stabilize.
        @test length(model.result.elbo_history) == 50
        @test isfinite(model.result.final_elbo)
        @test std(model.result.elbo_history[(end - 9):end]) < 1.0

        # Algorithm 1 parameter blocks and StatsAPI-compatible output.
        @test all(k -> haskey(model.result.variational_posterior, k), (:δ, :γ, :Σ))
        @test !haskey(model.result.variational_posterior, :μ)
        @test length(coef(model)) == 1
        @test length(stderror(model)) == 1
        @test size(confint(model)) == (1, 2)
        @test coeftable(model).method_type == :VI
    end
end

@testset "VMP argument validation" begin
    df = make_plr_DTL2025(20, 3, 1.0; alpha = 1.0)
    model = BDMLModel(df, :y, :d; model_type = :basic)

    @test_throws ArgumentError fit!(model, VMP(); n_iterations = 0)
    @test_throws ArgumentError fit!(model, VMP(); n_draws = 0)
    @test_throws ArgumentError fit!(model, VMP(); ν0 = 3.0)
    @test_throws ArgumentError fit!(model, VMP(); S0 = ones(3, 3))
    @test_throws ArgumentError fit!(model, VMP(); limit_stack_depth = 0)
    @test_throws ArgumentError fit!(model, VMP(); limit_stack_depth = -1)
end

@testset "VMP large model stack depth limit" begin
    Random.seed!(1234)
    # n=3000 is large enough to trigger StackOverflowError on default stack size
    df = make_plr_DTL2025(3000, 25, 1.0; alpha = 2.0)

    # Auto-default (n > 2000) should prevent overflow
    model_hier = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_hier, VMP(); n_iterations = 20, n_draws = 100, seed = 42)
    @test isfitted(model_hier)
    @test all(isfinite, model_hier.result.alpha_samples)
    @test abs(mean(model_hier.result.alpha_samples) - 2.0) < 0.5

    # Explicit limit should also work
    model_basic = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model_basic, VMP(); n_iterations = 20, n_draws = 100, seed = 42, limit_stack_depth = 100)
    @test isfitted(model_basic)
    @test all(isfinite, model_basic.result.alpha_samples)
end
