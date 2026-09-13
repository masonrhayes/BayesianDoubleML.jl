# VMP manual backend without RxInfer extension loaded
using BayesianDoubleML
using Distributions
using LinearAlgebra
using Random
using Statistics
using Test

@testset "VMP effective residual degrees of freedom" begin
    qΣ = InverseWishart(124.0, [4.0 0.8; 0.8 2.0])
    eigenvalues = collect(1.0:100.0)
    effective_df = BayesianDoubleML._vmp_effective_df(
        eigenvalues, qΣ, 1.0e-10, 1.0e-10
    )
    shrunk_df = BayesianDoubleML._vmp_effective_df(eigenvalues, qΣ, 1.0e6, 1.0e6)
    corrected_qΣ = BayesianDoubleML._vmp_adjust_covariance(qΣ, effective_df)

    raw_ν, raw_S = params(qΣ)
    corrected_ν, corrected_S = params(corrected_qΣ)
    raw_alpha_variance = det(raw_S) / (raw_S[2, 2]^2 * (raw_ν - 2))
    corrected_alpha_variance = det(corrected_S) /
        (corrected_S[2, 2]^2 * (corrected_ν - 2))

    @test effective_df ≈ length(eigenvalues) atol = 1.0e-8
    @test 0 < shrunk_df < effective_df
    @test corrected_ν ≈ raw_ν - effective_df
    @test mean(corrected_qΣ) ≈ mean(qΣ)
    @test corrected_alpha_variance / raw_alpha_variance ≈
        (raw_ν - 2) / (corrected_ν - 2)
end

@testset "VMP high-dimensional variance correction" begin
    n, p = 120, 100
    df = make_plr_DTL2025(n, p, 2.0; alpha = 2.0, rng = Xoshiro(42))
    model = BDMLModel(df, :y, :d; model_type = :basic)
    fit!(model, VMP(); n_iterations = 100, n_draws = 10_000, rng = Xoshiro(7))

    raw_ν, raw_S = params(model.result.posterior.Σ_vmp)
    corrected_ν, corrected_S = params(model.result.posterior.Σ)
    raw_variance = det(raw_S) / (raw_S[2, 2]^2 * (raw_ν - 2))
    corrected_variance = det(corrected_S) /
        (corrected_S[2, 2]^2 * (corrected_ν - 2))

    @test model.result.posterior.effective_df > 0.9p
    @test corrected_variance > 4raw_variance
    @test var(model.result.alpha_samples_standardized) ≈ corrected_variance rtol = 0.15
end

@testset "VMP default backend without RxInfer" begin
    df = make_plr_DTL2025(60, 4, 1.0; alpha = 1.0)
    model = BDMLModel(df, :y, :d; model_type = :basic)

    # Default VMP() now uses ManualCoordinateAscentVMP and works without RxInfer
    fit!(model, VMP(); n_iterations = 20, n_draws = 100, rng = Xoshiro(1))
    @test isfitted(model)
    @test model.result isa BDMLVMPResult
    @test model.result.backend == :manual_coordinate_ascent
    @test all(isfinite, model.result.alpha_samples)
    @test haskey(model.result.posterior, :Σ_vmp)
    @test 0 < model.result.posterior.effective_df < model.p

    corrected_ν, corrected_S = params(model.result.posterior.Σ)
    expected_variance = det(corrected_S) /
        (corrected_S[2, 2]^2 * (corrected_ν - 2))
    @test var(model.result.alpha_samples_standardized) ≈ expected_variance rtol = 0.35

    # Explicit RxInfer backend should raise informative error when extension not loaded
    model2 = BDMLModel(df, :y, :d; model_type = :basic)
    method_rx = VMP(; backend = RxInferVMP())
    @test_throws ErrorException fit!(model2, method_rx; n_iterations = 10)
    @test !isfitted(model2)
end
