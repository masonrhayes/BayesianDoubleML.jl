# VMP manual backend without RxInfer extension loaded
using BayesianDoubleML
using Random
using Test

@testset "VMP default backend without RxInfer" begin
    df = make_plr_DTL2025(60, 4, 1.0; alpha = 1.0)
    model = BDMLModel(df, :y, :d; model_type = :basic)

    # Default VMP() now uses ManualCoordinateAscentVMP and works without RxInfer
    fit!(model, VMP(); n_iterations = 20, n_draws = 100, rng = Xoshiro(1))
    @test isfitted(model)
    @test model.result isa BDMLVMPResult
    @test model.result.backend == :manual_coordinate_ascent
    @test all(isfinite, model.result.alpha_samples)

    # Explicit RxInfer backend should raise informative error when extension not loaded
    model2 = BDMLModel(df, :y, :d; model_type = :basic)
    method_rx = VMP(; backend = RxInferVMP())
    @test_throws ErrorException fit!(model2, method_rx; n_iterations = 10)
    @test !isfitted(model2)
end
