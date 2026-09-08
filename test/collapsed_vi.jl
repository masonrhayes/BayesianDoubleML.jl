using BayesianDoubleML
using Distributions
using LinearAlgebra
using LogDensityProblems
using Mooncake
using StableRNGs
using Statistics
using Test

const BDML = BayesianDoubleML

@testset "Collapsed marginal likelihood" begin
    df = make_plr_DTL2025(12, 5, 2.0; alpha = 0.7, rng = StableRNG(2))

    for model_kind in (:basic, :hier)
        model = BDMLModel(df, :y, :d; model_type = model_kind)
        collapsed = BDML.CollapsedBDML(model)
        sigma_U, sigma_V, rho = 0.8, 1.2, 0.35
        s2d, s2g = model_kind === :hier ? (0.7, 1.4) : (25.0, 25.0)
        Sigma = [
            sigma_U^2 rho * sigma_U * sigma_V
            rho * sigma_U * sigma_V sigma_V^2
        ]

        coefficient_covariance = Diagonal(vcat(fill(s2d, collapsed.p), fill(s2g, collapsed.p)))
        design = [
            model.X zeros(collapsed.n, collapsed.p)
            zeros(collapsed.n, collapsed.p) model.X
        ]
        covariance = kron(Sigma, I(collapsed.n)) +
            design * coefficient_covariance * design'
        brute_force = logpdf(
            MvNormal(zeros(2 * collapsed.n), Symmetric(covariance)),
            vcat(model.Y, model.D),
        )
        actual = BDML.log_marginal_collapsed(
            collapsed, sigma_U, sigma_V, rho, s2d, s2g,
        )

        @test actual ≈ brute_force atol = 1.0e-10
    end
end

@testset "Collapsed rank-deficient target" begin
    rng = StableRNG(3)
    model = BDMLModel(randn(rng, 8), randn(rng, 8), randn(rng, 8, 10))
    collapsed = BDML.CollapsedBDML(model)
    problem = BDML.CollapsedBDMLProblem(collapsed)

    @test collapsed.rank < collapsed.p
    @test isfinite(LogDensityProblems.logdensity(problem, zeros(3)))
end

@testset "CollapsedVI fits both models and families" begin
    df = make_plr_DTL2025(100, 10, 2.0; alpha = 0.8, rng = StableRNG(301))

    for model_kind in (:basic, :hier), fullrank in (false, true)
        model = BDMLModel(df, :y, :d; model_type = model_kind)
        fit!(
            model,
            CollapsedVI(; fullrank);
            n_iterations = 300,
            n_draws = 300,
            rng = StableRNG(9),
            show_progress = false,
        )

        @test isfitted(model)
        @test model.result isa BDMLVIResult
        @test model.result.model_type == model_kind
        @test model.result.vi_method == :collapsed
        @test model.result.n_iterations == 300
        @test isfinite(model.result.final_elbo)
        @test all(isfinite, model.result.alpha_samples)
        @test abs(mean(model.result.alpha_samples) - 0.8) < 0.5
        if fullrank
            @test parent(model.result.variational_posterior.scale) isa Matrix
        end
    end
end

@testset "CollapsedVI Mooncake direct AD" begin
    df = make_plr_DTL2025(40, 10, 2.0; alpha = 0.7, rng = StableRNG(7))
    model = BDMLModel(df, :y, :d; model_type = :hier)
    fit!(
        model,
        CollapsedVI(; ad_backend = AutoMooncake);
        n_iterations = 10,
        n_draws = 50,
        rng = StableRNG(8),
        show_progress = false,
    )

    @test isfinite(model.result.final_elbo)
    @test all(isfinite, model.result.alpha_samples)
end
