# Variational Message Passing tests
using BayesianDoubleML
using RxInfer
using Random
using Statistics
using Test

@testset "VMP extension activation" begin
    @test Base.get_extension(BayesianDoubleML, :BayesianDoubleMLRxInferExt) !== nothing
end

@testset "VMP basic and hierarchical (RxInfer)" begin
    Random.seed!(8128)
    df = make_plr_DTL2025(100, 10, 1.0; alpha = 2.0)

    for model_type in (:basic, :hier)
        model = BDMLModel(df, :y, :d; model_type)
        method = VMP(; backend = RxInferVMP())
        fit!(model, method; n_iterations = 50, n_draws = 300, rng = Xoshiro(42))

        @test isfitted(model)
        @test model.result isa BDMLVMPResult
        @test model.result.backend == :rxinfer
        @test model.result.diagnostic_kind == :elbo
        @test model.result.n_iterations == 50
        @test length(model.result.alpha_samples) == 300
        @test all(isfinite, model.result.alpha_samples)
        @test abs(mean(model.result.alpha_samples) - 2.0) < 0.5
        @test model.result.converged

        @test length(model.result.diagnostic_history) == 50
        @test isfinite(model.result.final_diagnostic)
        @test std(model.result.diagnostic_history[(end - 9):end]) < 1.0

        @test all(k -> haskey(model.result.posterior, k), (:δ, :γ, :Σ))
        @test !haskey(model.result.posterior, :μ)
        @test length(coef(model)) == 1
        @test length(stderror(model)) == 1
        @test size(confint(model)) == (1, 2)
        @test coeftable(model).method_type == :VMP
    end
end

@testset "VMP argument validation" begin
    df = make_plr_DTL2025(20, 3, 1.0; alpha = 1.0)

    @test_throws ArgumentError fit!(BDMLModel(df, :y, :d; model_type = :basic), VMP(); n_iterations = 0)
    @test_throws ArgumentError fit!(BDMLModel(df, :y, :d; model_type = :basic), VMP(); n_draws = 0)
    @test_throws ArgumentError VMP(; ν0 = 3.0)
    @test_throws ArgumentError VMP(; S0 = ones(3, 3))
    @test_throws ArgumentError RxInferVMP(; limit_stack_depth = 0)
    @test_throws ArgumentError RxInferVMP(; limit_stack_depth = -1)

    # Legacy keywords must be rejected (MethodError because _fit_vmp does not accept them)
    @test_throws MethodError fit!(BDMLModel(df, :y, :d; model_type = :basic), VMP(); backend = :rxinfer)
    @test_throws MethodError fit!(BDMLModel(df, :y, :d; model_type = :basic), VMP(); seed = 42)
    @test_throws MethodError fit!(BDMLModel(df, :y, :d; model_type = :basic), VMP(); showprogress = false)
end

@testset "VMP manual-coordinate-ascent backend" begin
    Random.seed!(9182)
    df = make_plr_DTL2025(100, 10, 1.0; alpha = 2.0)

    for model_type in (:basic, :hier)
        model = BDMLModel(df, :y, :d; model_type)
        method = VMP(; backend = ManualCoordinateAscentVMP())
        fit!(model, method; n_iterations = 50, n_draws = 300, rng = Xoshiro(42))

        @test isfitted(model)
        @test model.result isa BDMLVMPResult
        @test model.result.backend == :manual_coordinate_ascent
        @test model.result.diagnostic_kind == :parameter_change
        @test all(isfinite, model.result.alpha_samples)
        @test abs(mean(model.result.alpha_samples) - 2.0) < 0.5
        @test length(model.result.diagnostic_history) == 50
        @test all(isfinite, model.result.diagnostic_history)
        @test all(k -> haskey(model.result.posterior, k), (:δ, :γ, :Σ))
    end
end

@testset "VMP backends agree (basic)" begin
    Random.seed!(4401)
    df = make_plr_DTL2025(80, 6, 1.0; alpha = 1.5)
    rx_model = BDMLModel(df, :y, :d; model_type = :basic)
    ss_model = BDMLModel(df, :y, :d; model_type = :basic)

    fit!(rx_model, VMP(; backend = RxInferVMP()); n_iterations = 40, n_draws = 1000, rng = Xoshiro(1))
    fit!(ss_model, VMP(; backend = ManualCoordinateAscentVMP()); n_iterations = 40, n_draws = 1000, rng = Xoshiro(1))

    rx = rx_model.result.posterior
    ss = ss_model.result.posterior
    @test isapprox(mean(rx[:δ]), mean(ss.δ); rtol = 1.0e-3, atol = 1.0e-5)
    @test isapprox(mean(rx[:γ]), mean(ss.γ); rtol = 1.0e-3, atol = 1.0e-5)
    @test isapprox(mean(rx[:Σ]), mean(ss.Σ); rtol = 1.0e-2, atol = 1.0e-5)
    @test isapprox(mean(rx_model.result.alpha_samples), mean(ss_model.result.alpha_samples); rtol = 2.0e-2)
end

@testset "VMP backends agree (hierarchical)" begin
    Random.seed!(4402)
    df = make_plr_DTL2025(80, 6, 1.0; alpha = 1.5)
    rx_model = BDMLModel(df, :y, :d; model_type = :hier)
    ss_model = BDMLModel(df, :y, :d; model_type = :hier)

    fit!(rx_model, VMP(; backend = RxInferVMP()); n_iterations = 40, n_draws = 1000, rng = Xoshiro(1))
    fit!(ss_model, VMP(; backend = ManualCoordinateAscentVMP()); n_iterations = 40, n_draws = 1000, rng = Xoshiro(1))

    rx = rx_model.result.posterior
    ss = ss_model.result.posterior
    @test isapprox(mean(rx[:δ]), mean(ss.δ); rtol = 5.0e-3, atol = 1.0e-5)
    @test isapprox(mean(rx[:γ]), mean(ss.γ); rtol = 5.0e-3, atol = 1.0e-5)
    @test isapprox(mean(rx[:Σ]), mean(ss.Σ); rtol = 1.0e-2, atol = 1.0e-5)
    @test isapprox(mean(rx_model.result.alpha_samples), mean(ss_model.result.alpha_samples); rtol = 2.0e-2)

    # Hierarchical-specific posterior blocks
    @test haskey(rx, :τ_δ)
    @test haskey(rx, :τ_γ)
    @test haskey(ss, :τ_δ)
    @test haskey(ss, :τ_γ)
end

@testset "VMP RNG reproducibility" begin
    df = make_plr_DTL2025(60, 4, 1.0; alpha = 1.0)
    model = BDMLModel(df, :y, :d; model_type = :basic)
    method = VMP(; backend = ManualCoordinateAscentVMP())

    fit!(model, method; n_iterations = 20, n_draws = 200, rng = MersenneTwister(123))
    s1 = copy(model.result.alpha_samples)

    model2 = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model2, method; n_iterations = 20, n_draws = 200, rng = MersenneTwister(123))
    s2 = model2.result.alpha_samples

    @test s1 == s2

    model3 = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model3, method; n_iterations = 20, n_draws = 200, rng = MersenneTwister(124))
    @test s1 != model3.result.alpha_samples
end

@testset "VMP large model stack depth limit" begin
    Random.seed!(1234)
    df = make_plr_DTL2025(3000, 25, 1.0; alpha = 2.0)

    # Auto-default (n > 2000) should prevent overflow
    model_hier = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(model_hier, VMP(; backend = RxInferVMP()); n_iterations = 20, n_draws = 100, rng = Xoshiro(42))
    @test isfitted(model_hier)
    @test all(isfinite, model_hier.result.alpha_samples)
    @test abs(mean(model_hier.result.alpha_samples) - 2.0) < 0.5

    # Explicit limit should also work
    model_basic = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model_basic, VMP(; backend = RxInferVMP(; limit_stack_depth = 100)); n_iterations = 20, n_draws = 100, rng = Xoshiro(42))
    @test isfitted(model_basic)
    @test all(isfinite, model_basic.result.alpha_samples)
end

@testset "VMP summary and show" begin
    df = make_plr_DTL2025(50, 5, 1.0; alpha = 1.0)

    model_rx = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model_rx, VMP(; backend = RxInferVMP()); n_iterations = 10, n_draws = 50, rng = Xoshiro(1))
    @test occursin("VMP", sprint(summary, model_rx.result))

    model_man = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model_man, VMP(; backend = ManualCoordinateAscentVMP()); n_iterations = 10, n_draws = 50, rng = Xoshiro(1))
    @test occursin("VMP", sprint(summary, model_man.result))

    ct_rx = coeftable(model_rx.result)
    @test ct_rx.method_type == :VMP

    ct_man = coeftable(model_man.result)
    @test ct_man.method_type == :VMP
end
