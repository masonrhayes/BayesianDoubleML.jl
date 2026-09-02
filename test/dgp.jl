using BayesianDoubleML
using LinearAlgebra
using Random
using Statistics
using Test

@testset "APD2022 binary-treatment DGP" begin
    df = make_irm_APD2022(200, 8; rng = MersenneTwister(1))
    @test size(df) == (200, 10)
    @test names(df)[1:8] == ["X$i" for i in 1:8]
    @test names(df)[9:10] == ["y", "d"]
    @test all(df.d .∈ Ref([0.0, 1.0]))
    @test length(unique(df.d)) == 2
    @test BayesDRModel(df, :y, :d) isa BayesDRModel

    repeated = make_irm_APD2022(200, 8; rng = MersenneTwister(1))
    @test df == repeated
    @test df == make_irm_APD2022(MersenneTwister(1); n = 200, p = 8)
    @test_throws ArgumentError make_irm_APD2022(20, 4)
    @test_throws ArgumentError make_irm_APD2022(20, 5; scenario = :unknown)
end

@testset "DGP RNG-first interface" begin
    positional = make_plr_DTL2025(
        100, 8, 2.0; alpha = 0.5, rng = MersenneTwister(9),
    )
    rng_first = make_plr_DTL2025(
        MersenneTwister(9); n = 100, p = 8, sigma_epsilon = 2.0, alpha = 0.5,
    )
    @test rng_first == positional
end

@testset "APD2022 causal effect construction" begin
    for scenario in (:linear, :nonlinear)
        untreated = make_irm_APD2022(
            300, 5; scenario, alpha = 0.0, rng = MersenneTwister(2),
        )
        treated = make_irm_APD2022(
            300, 5; scenario, alpha = 2.0, rng = MersenneTwister(2),
        )
        @test treated.d == untreated.d
        @test treated.y - untreated.y ≈ 2 .* treated.d
    end
end

@testset "APD2022 equicorrelated covariates" begin
    df = make_irm_APD2022(20_000, 5; rng = MersenneTwister(3))
    covariance = cov(Matrix(df[:, 1:5]))
    @test diag(covariance) ≈ ones(5) atol = 0.04
    off_diagonal = [covariance[i, j] for i in 1:5 for j in 1:5 if i != j]
    @test mean(off_diagonal) ≈ 0.3 atol = 0.03
end

@testset "APD2022 continuous-treatment DGP" begin
    df = make_er_APD2022(200, 8; rng = MersenneTwister(4))
    @test size(df) == (200, 10)
    @test names(df)[1:8] == ["X$i" for i in 1:8]
    @test names(df)[9:10] == ["y", "d"]
    @test length(unique(df.d)) > 2
    @test all(isfinite, Matrix(df))
    @test BayesDRModel(df, :y, :d).treatment_type == :continuous
    @test df == make_er_APD2022(200, 8; rng = MersenneTwister(4))
    @test df == make_er_APD2022(MersenneTwister(4); n = 200, p = 8)
    @test size(make_er_APD2022(MersenneTwister(5))) == (200, 202)
    @test_throws ArgumentError make_er_APD2022(0, 3)
    @test_throws ArgumentError make_er_APD2022(20, 2)
    @test_throws ArgumentError make_er_APD2022(; cubic_coefficient = Inf)
    @test_throws ArgumentError make_er_APD2022(; quadratic_coefficient = NaN)
end

@testset "APD2022 continuous response customization" begin
    no_effect = make_er_APD2022(
        200, 3; cubic_coefficient = 0.0, quadratic_coefficient = 0.0,
        rng = MersenneTwister(6),
    )
    custom = make_er_APD2022(
        200, 3; cubic_coefficient = 0.2, quadratic_coefficient = -0.3,
        rng = MersenneTwister(6),
    )
    rng_first = make_er_APD2022(
        MersenneTwister(6); n = 200, p = 3,
        cubic_coefficient = 0.2, quadratic_coefficient = -0.3,
    )
    @test custom[:, 1:3] == no_effect[:, 1:3]
    @test custom.d == no_effect.d
    @test custom.y - no_effect.y ≈ 0.2 .* custom.d .^ 3 .- 0.3 .* custom.d .^ 2
    @test rng_first == custom
end

@testset "APD2022 continuous paper equations" begin
    df = make_er_APD2022(30_000, 3; rng = MersenneTwister(5))
    X = Matrix(df[:, 1:3])
    covariance = cov(X)
    @test diag(covariance) ≈ ones(3) atol = 0.03
    off_diagonal = [covariance[i, j] for i in 1:3 for j in 1:3 if i != j]
    @test mean(off_diagonal) ≈ 0.3 atol = 0.02

    treatment_mean = @. 0.6 * df.X1 + 0.6 * df.X2 +
        exp(0.65 * abs(df.X1)) - 0.8 * df.X3^2
    treatment_error = df.d - treatment_mean
    outcome_mean = @. 5 + 0.05 * df.d^3 - 0.1 * df.d^2 + 0.6 * df.X1 +
        0.4 * exp(df.X1) + log(0.65 * abs(df.X2)) + 0.5 * (1 + df.X3)^2
    outcome_error = df.y - outcome_mean

    @test mean(treatment_error) ≈ 0.0 atol = 0.02
    @test var(treatment_error) ≈ 1.0 atol = 0.03
    @test mean(outcome_error) ≈ 0.0 atol = 0.02
    @test var(outcome_error) ≈ 1.0 atol = 0.03
    @test maximum(abs, cor(hcat(X, treatment_error, outcome_error))[1:3, 4:5]) < 0.025
    @test cor(treatment_error, outcome_error) ≈ 0.0 atol = 0.02
end

@testset "LML2025 continuous-exposure DGP" begin
    df = make_plr_LML2025(MersenneTwister(4))
    @test size(df) == (40, 42)
    @test names(df)[1:40] == ["X$i" for i in 1:40]
    @test names(df)[41:42] == ["y", "d"]
    @test df == make_plr_LML2025(MersenneTwister(4))
    @test size(make_plr_LML2025(MersenneTwister(5); n = 12, p = 7)) == (12, 9)
    @test BDMLModel(df, :y, :d) isa BDMLBasicModel

    @test_throws ArgumentError make_plr_LML2025(; n = 0)
    @test_throws ArgumentError make_plr_LML2025(; p = 6)
    @test_throws ArgumentError make_plr_LML2025(; treatment = :unknown)
    @test_throws ArgumentError make_plr_LML2025(; alpha = Inf)
end


@testset "LML2025 treatment effect customization" begin
    for treatment in (:continuous, :binary)
        no_effect = make_plr_LML2025(
            MersenneTwister(7); n = 200, p = 10, treatment, alpha = 0.0,
        )
        custom = make_plr_LML2025(
            MersenneTwister(7); n = 200, p = 10, treatment, alpha = 2.5,
        )
        @test custom[:, 1:10] == no_effect[:, 1:10]
        @test custom.d == no_effect.d
        @test custom.y - no_effect.y ≈ 2.5 .* custom.d
    end
end

@testset "LML2025 paper equations" begin
    df = make_plr_LML2025(MersenneTwister(6); n = 30_000, p = 7)
    X = Matrix(df[:, 1:7])
    covariance = cov(X)

    @test diag(covariance) ≈ ones(7) atol = 0.03
    off_diagonal = [covariance[i, j] for i in 1:7 for j in 1:7 if i != j]
    @test mean(off_diagonal) ≈ 0.05 atol = 0.015

    treatment_error = df.d .- (0.45 .* df.X1 .+ 0.9 .* df.X2 .- 0.4 .* df.X5)
    outcome_error = df.y .- (
        df.d .+ 0.5 .* df.X1 .+ df.X3 .- 0.1 .* df.X4 .- 0.2 .* df.X7
    )
    @test mean(treatment_error) ≈ 0.0 atol = 0.02
    @test var(treatment_error) ≈ 1.0 atol = 0.03
    @test mean(outcome_error) ≈ 0.0 atol = 0.02
    @test var(outcome_error) ≈ 1.0 atol = 0.03
    @test maximum(abs, cor(hcat(X, treatment_error, outcome_error))[1:7, 8:9]) < 0.025
    @test cor(treatment_error, outcome_error) ≈ 0.0 atol = 0.02
end

@testset "LML2025 binary-treatment DGP" begin
    df = make_plr_LML2025(MersenneTwister(7); n = 200, p = 10, treatment = :binary)
    @test size(df) == (200, 12)
    @test names(df)[1:10] == ["X$i" for i in 1:10]
    @test names(df)[11:12] == ["y", "d"]
    @test all(df.d .∈ Ref([0.0, 1.0]))
    @test length(unique(df.d)) == 2
    @test df == make_plr_LML2025(
        MersenneTwister(7); n = 200, p = 10, treatment = :binary,
    )
end

@testset "LML2025 binary paper equations" begin
    df = make_plr_LML2025(
        MersenneTwister(8); n = 30_000, p = 7, treatment = :binary,
    )
    X = Matrix(df[:, 1:7])
    covariance = cov(X)

    @test diag(covariance) ≈ ones(7) atol = 0.03
    off_diagonal = [covariance[i, j] for i in 1:7 for j in 1:7 if i != j]
    @test mean(off_diagonal) ≈ 0.3 atol = 0.02

    linear_predictor = 0.3 .* df.X1 .+ 0.2 .* df.X2 .- 0.4 .* df.X5
    propensity = @. inv(1 + exp(-linear_predictor))
    treatment_error = df.d .- propensity
    outcome_error = df.y .- (
        df.d .+ 0.5 .* df.X1 .+ df.X3 .- 0.1 .* df.X4 .- 0.2 .* df.X7
    )
    @test mean(treatment_error) ≈ 0.0 atol = 0.01
    @test maximum(abs, cor(hcat(X, treatment_error))[1:7, 8]) < 0.025
    @test mean(outcome_error) ≈ 0.0 atol = 0.02
    @test var(outcome_error) ≈ 1.0 atol = 0.03
    @test cor(treatment_error, outcome_error) ≈ 0.0 atol = 0.02
end
