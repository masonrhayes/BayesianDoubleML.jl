using BayesianDoubleML
using DataFrames
using LinearAlgebra
using Random
using Statistics
using Test

@testset "BayesDR construction and validation" begin
    Y = [1.0, 2.0, 3.0, 4.0]
    T = [0.0, 1.0, 0.0, 1.0]
    X = [1.0 4.0; 2.0 3.0; 3.0 2.0; 4.0 1.0]
    model = BayesDRModel(Y, T, X)

    @test model_type(model) == :bayes_dr
    @test model.T == T
    @test model.Y == Y
    @test model.treatment_type == :binary
    @test nobs(model) == 4
    @test ncovariates(model) == 2
    @test all(abs.(vec(mean(model.X; dims = 1))) .< 1.0e-12)
    @test !isfitted(model)

    df = DataFrame(y = Y, treatment = T, x1 = X[:, 1], x2 = X[:, 2])
    @test BayesDRModel(df, :y, :treatment) isa BayesDRModel
    @test_throws ArgumentError BayesDRModel(Y, fill(0.0, 4), X)
    continuous = BayesDRModel(Y, [0.0, 1.0, 0.5, 1.0], X)
    @test continuous.treatment_type == :continuous
    @test_throws ArgumentError BayesDRModel(
        Y, [0.0, 1.0, 0.5, 1.0], X; treatment_type = :binary,
    )
    @test_throws ArgumentError BayesDRModel(Y, T, X; treatment_type = :continuous)
    @test_throws ArgumentError BayesDRModel(Y, T, hcat(X[:, 1], ones(4)))
    @test_throws DimensionMismatch BayesDRModel(Y, T[1:3], X)
end

@testset "BayesDR AIPW contributions" begin
    Y = [1.0, 3.0, 2.0, 4.0]
    T = [0.0, 1.0, 0.0, 1.0]
    propensity = [0.2 0.4 0.6 0.8; 0.3 0.5 0.7 0.9]
    outcome0 = [1.0 1.0 2.0 2.0; 1.0 1.0 2.0 2.0]
    outcome1 = outcome0 .+ 2.0
    contributions = BayesianDoubleML._aipw_contributions(
        Y, T, propensity, outcome0, outcome1,
    )
    @test contributions ≈ fill(2.0, 2, 4)
end

@testset "BayesDR sampler building blocks" begin
    treatment = [-1.0, 0.0, 2.0]
    @test BayesianDoubleML._treatment_basis(treatment, 3) ==
        hcat(treatment, treatment .^ 2, treatment .^ 3)

    x = [1.0, 2.0]
    residual = [0.5, -1.0]
    sigma2 = 2.0
    slab_variance = 4.0
    inclusion_probability = 0.25
    posterior_mean, posterior_variance, log_probability =
        BayesianDoubleML._inclusion_conditional(
        x, sum(abs2, x), residual, sigma2, slab_variance,
        inclusion_probability,
    )
    precision = sum(abs2, x) + inv(slab_variance)
    @test posterior_mean ≈ dot(x, residual) / precision
    @test posterior_variance ≈ sigma2 / precision
    log_p0 = log1p(-inclusion_probability)
    log_p1 = log(inclusion_probability) +
        0.5 * (log(posterior_variance) - log(sigma2 * slab_variance)) +
        0.5 * posterior_mean^2 / posterior_variance
    @test exp(log_probability) ≈ exp(log_p1) / (exp(log_p0) + exp(log_p1))

    posterior(chain, offset) = BayesianDoubleML.BayesDRNuisancePosterior(
        reshape([offset + 1.0, offset + 2.0], 1, 2),
        BitMatrix(reshape([true], 1, 1)),
        [offset + 3.0],
        [offset + 4.0],
        [0.5],
        [chain],
    )
    combined = BayesianDoubleML._combine_posteriors([posterior(1, 0), posterior(2, 10)])
    @test combined.coefficients == [1.0 2.0; 11.0 12.0]
    @test combined.chain_id == [1, 2]

    design = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
    response = [0.0, 1.0, 2.0]
    coefficients = zeros(2)
    BayesianDoubleML._sample_unpenalized!(
        MersenneTwister(4), coefficients, design,
        cholesky(Symmetric(design' * design)), response, zeros(3), 1.0,
        zeros(3), zeros(2),
    )
    @test length(coefficients) == 2
    @test all(isfinite, coefficients)
end

@testset "BayesDR method validation" begin
    method = BayesDRMCMC()
    @test method isa BayesDRMCMCMethod
    @test BayesianDoubleML.uses_sampling(method)
    @test !BayesianDoubleML.supports_subsampling(method)
    @test BayesianDoubleML.default_n_samples(method) == 1000
    @test_throws ArgumentError BayesDRMCMC(; propensity_bounds = (0.9, 0.1))
    @test_throws ArgumentError BayesDRMCMC(; slab_shape = 0.0)
    @test_throws ArgumentError BayesDRMCMC(; curve_degree = 0)
    @test_throws ArgumentError BayesDRMCMC(; density_ratio_bounds = (1.0, 0.5))
    @test_throws ArgumentError BayesianDoubleML.BayesDRSamplingPlan(1, 0, 1, 1, 2, 0.95)
    @test_throws ArgumentError BayesianDoubleML.BayesDRSamplingPlan(2, 0, 1, 1, 2, 1.0)

    model = BayesDRModel(
        [0.0, 1.0, 0.5, 1.5], [0.0, 1.0, 0.0, 1.0],
        [0.0 1.0; 1.0 0.0; 2.0 3.0; 3.0 2.0],
    )
    @test_throws ArgumentError fit!(
        model, method;
        n_samples = 2, n_burn = 0, n_chains = 1, n_boot = 2,
        treatment_grid = [0.0], rng = MersenneTwister(3),
    )
end

@testset "BayesDR nuisance diagnostics" begin
    function nuisance_posterior(rng; n_iterations = 100, shift = 0.0, n_chains = 2)
        coefficients = Matrix{Float64}(undef, n_iterations * n_chains, 2)
        inclusion = falses(n_iterations * n_chains, 1)
        residual_variance = Vector{Float64}(undef, n_iterations * n_chains)
        slab_variance = similar(residual_variance)
        inclusion_probability = similar(residual_variance)
        chain_id = repeat(1:n_chains; inner = n_iterations)
        for chain in 1:n_chains
            rows = ((chain - 1) * n_iterations + 1):(chain * n_iterations)
            coefficients[rows, :] .= randn(rng, n_iterations, 2)
            chain == 2 && (coefficients[rows, 1] .+= shift)
            inclusion[rows, 1] .= rand(rng, n_iterations) .< 0.4
            residual_variance[rows] .= exp.(0.1 .* randn(rng, n_iterations))
            slab_variance[rows] .= exp.(0.1 .* randn(rng, n_iterations))
            inclusion_probability[rows] .= rand(rng, n_iterations)
        end
        return BayesianDoubleML.BayesDRNuisancePosterior(
            coefficients, inclusion, residual_variance, slab_variance,
            inclusion_probability, chain_id,
        )
    end

    mixed = nuisance_posterior(MersenneTwister(12))
    @test ess(mixed) > 0
    @test isfinite(rhat(mixed))
    @test mcse(mixed) > 0
    @test chain_info(mixed) ==
        (n_chains = 2, n_samples_per_chain = 100, total_samples = 200)

    shifted = nuisance_posterior(MersenneTwister(12); shift = 3.0)
    @test rhat(shifted) > 1.1

    single = nuisance_posterior(MersenneTwister(12); n_chains = 1)
    @test ismissing(rhat(single))
end

@testset "BayesDR continuous treatment" begin
    rng = MersenneTwister(2023)
    n = 50
    X = randn(rng, n, 4)
    T = 0.7 .* X[:, 1] .- 0.4 .* X[:, 2] .+ randn(rng, n)
    response_curve(t) = 1.0 + 0.6 * t - 0.15 * t^2 + 0.04 * t^3
    Y = response_curve.(T) .+ 0.8 .* X[:, 1] .+ 0.4 .* randn(rng, n)
    grid = [-1.0, 0.0, 1.0]

    model = BayesDRModel(Y, T, X)
    @test model.treatment_type == :continuous
    design = BayesianDoubleML._continuous_design(model, BayesDRMCMC(), grid)
    standardized_treatment = (T .- mean(T)) ./ std(T)
    @test design.treatment ≈ standardized_treatment
    @test design.treatment_basis ≈
        BayesianDoubleML._treatment_basis(standardized_treatment, 3)
    @test design.grid_design[:, 1] == ones(length(grid))
    fit!(
        model, BayesDRMCMC();
        n_samples = 8,
        n_burn = 5,
        n_chains = 1,
        n_boot = 5,
        treatment_grid = grid,
        rng = MersenneTwister(7),
    )

    result = model.result
    @test result isa BayesDRCurveResult
    @test result.treatment_grid == grid
    @test size(result.posterior_curves) == (8, 3)
    @test size(result.bootstrap_curves) == (5, 3)
    @test size(result.confidence_interval) == (3, 2)
    @test size(vcov(result)) == (3, 3)
    @test coef(result) == result.estimate
    @test stderror(result) == result.standard_error
    @test confint(result) == result.confidence_interval
    curve = exposure_response_curve(result)
    @test names(curve) == ["treatment", "estimate", "standard_error", "lower", "upper"]
    @test curve.treatment == grid
    @test curve.estimate == result.estimate
    @test curve.standard_error == result.standard_error
    @test hcat(curve.lower, curve.upper) == result.confidence_interval
    @test exposure_response_curve(model) == curve
    curve90 = exposure_response_curve(result; level = 0.9)
    @test hcat(curve90.lower, curve90.upper) == confint(result; level = 0.9)
    derivative = average_derivative(result)
    posterior_derivatives =
        (result.posterior_curves[:, 3] .- result.posterior_curves[:, 1]) ./ 2
    bootstrap_derivatives =
        (result.bootstrap_curves[:, 3] .- result.bootstrap_curves[:, 1]) ./ 2
    @test derivative.estimate ≈ (result.estimate[3] - result.estimate[1]) / 2
    @test derivative.naive_variance ≈ var(bootstrap_derivatives)
    @test derivative.posterior_variance ≈ var(posterior_derivatives)
    @test derivative.standard_error^2 ≈
        derivative.naive_variance + derivative.posterior_variance
    @test derivative.treatment_interval == (-1.0, 1.0)
    @test average_derivative(model) == derivative
    lower_derivative = average_derivative(result; treatment_interval = (-1.0, 0.0))
    @test lower_derivative.estimate ≈ result.estimate[2] - result.estimate[1]
    derivative90 = average_derivative(result; level = 0.9)
    @test derivative90.level == 0.9
    @test derivative90.confidence_interval != derivative.confidence_interval
    @test_throws ArgumentError average_derivative(
        result; treatment_interval = (1.0, -1.0),
    )
    @test_throws ArgumentError average_derivative(
        result; treatment_interval = (-0.5, 1.0),
    )
    @test all(isfinite, result.estimate)
    @test all(isfinite, result.standard_error)
    @test maximum(abs.(result.estimate .- response_curve.(grid))) < 1.0
    @test 0 <= result.density_ratio_clipped_fraction <= 1
    @test coeftable(result).interval_type == :confidence
    @test_throws ErrorException extract_alpha(result)
    @test_throws ErrorException credible_interval(result)
    summary_buffer = IOBuffer()
    @test_nowarn summary(summary_buffer, result)
    @test occursin("Average Derivative Diagnostic", String(take!(summary_buffer)))
    model_summary_buffer = IOBuffer()
    @test_nowarn summary(model_summary_buffer, model)
    @test occursin(
        "Average Derivative Diagnostic", String(take!(model_summary_buffer)),
    )
    @test_nowarn show(IOBuffer(), result)

    repeated = BayesDRModel(Y, T, X)
    fit!(
        repeated, BayesDRMCMC();
        n_samples = 8,
        n_burn = 5,
        n_chains = 1,
        n_boot = 5,
        treatment_grid = grid,
        rng = MersenneTwister(7),
    )
    @test repeated.result.posterior_curves == result.posterior_curves
    @test repeated.result.bootstrap_curves == result.bootstrap_curves
end

@testset "BayesDR summary curve plot" begin
    rng = MersenneTwister(2024)
    n = 50
    X = randn(rng, n, 4)
    T = 0.7 .* X[:, 1] .- 0.4 .* X[:, 2] .+ randn(rng, n)
    Y = @. 1.0 + 0.6 * T + 0.8 * X[:, 1] + 0.4 * randn(rng)
    grid = [-1.0, 0.0, 1.0]

    model = BayesDRModel(Y, T, X)
    fit!(
        model, BayesDRMCMC();
        n_samples = 8,
        n_burn = 5,
        n_chains = 1,
        n_boot = 5,
        treatment_grid = grid,
        rng = MersenneTwister(7),
    )

    plotted = IOBuffer()
    @test_nowarn summary(plotted, model; show_curve = true)
    @test occursin("Exposure-Response Curve E[Y(t)]", String(take!(plotted)))

    unplotted = IOBuffer()
    @test_nowarn summary(unplotted, model)
    @test !occursin("Exposure-Response Curve E[Y(t)]", String(take!(unplotted)))

    result_plotted = IOBuffer()
    @test_nowarn summary(result_plotted, model.result; show_curve = true)
    @test occursin("Exposure-Response Curve E[Y(t)]", String(take!(result_plotted)))
end

@testset "BayesDR end-to-end" begin
    rng = MersenneTwister(2022)
    n = 80
    X = randn(rng, n, 4)
    propensity = 1.0 ./ (1.0 .+ exp.(-(0.6 .* X[:, 1] .- 0.3 .* X[:, 2])))
    T = Float64.(rand(rng, n) .< propensity)
    true_ate = 0.75
    Y = true_ate .* T .+ 0.7 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ 0.5 .* randn(rng, n)

    model = BayesDRModel(Y, T, X)
    fit!(
        model, BayesDRMCMC();
        n_samples = 60,
        n_burn = 40,
        n_chains = 2,
        n_boot = 50,
        rng = MersenneTwister(99),
    )

    @test isfitted(model)
    @test model.result isa BayesDRResult
    result = model.result
    @test length(result.posterior_effects) == 120
    @test length(result.bootstrap_estimates) == 50
    @test result.standard_error^2 ≈ result.naive_variance + result.posterior_variance
    @test all((0.01 .<= result.propensity_mean) .& (result.propensity_mean .<= 0.99))
    @test 0 <= result.propensity_clipped_fraction <= 1
    @test abs(result.estimate - true_ate) < 0.35
    @test coef(model) == [result.estimate]
    @test stderror(model) == [result.standard_error]
    @test vcov(model)[1, 1] ≈ result.standard_error^2
    @test confint(model) == reshape(collect(result.confidence_interval), 1, 2)
    @test size(confint(result; level = 0.9)) == (1, 2)
    @test coeftable(model).interval_type == :confidence
    @test_throws ArgumentError exposure_response_curve(result)
    @test_throws ArgumentError average_derivative(result)
    @test_throws ErrorException credible_interval(model)
    @test_throws ErrorException credible_interval(result)
    @test_nowarn summary(IOBuffer(), result)
    @test_nowarn show(IOBuffer(), result)
    @test ess(model) > 0
    @test isfinite(rhat(model))
    @test mcse(model) > 0
    @test chain_info(model) ==
        (n_chains = 2, n_samples_per_chain = 60, total_samples = 120)
    @test length(vec(mean(result.treatment_posterior.inclusion; dims = 1))) == 4

    repeated = BayesDRModel(Y, T, X)
    fit!(
        repeated, BayesDRMCMC();
        n_samples = 60,
        n_burn = 40,
        n_chains = 2,
        n_boot = 50,
        rng = MersenneTwister(99),
    )
    @test repeated.result.posterior_effects == result.posterior_effects
    @test repeated.result.bootstrap_estimates == result.bootstrap_estimates
end

@testset "BayesDR RNG-first fit!" begin
    rng = MersenneTwister(2022)
    n = 80
    X = randn(rng, n, 4)
    propensity = 1.0 ./ (1.0 .+ exp.(-(0.6 .* X[:, 1] .- 0.3 .* X[:, 2])))
    T = Float64.(rand(rng, n) .< propensity)
    Y = 0.75 .* T .+ 0.7 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ 0.5 .* randn(rng, n)
    fit_kwargs = (n_samples = 60, n_burn = 40, n_chains = 2, n_boot = 50)

    reference = BayesDRModel(Y, T, X)
    fit!(reference, BayesDRMCMC(); fit_kwargs..., rng = MersenneTwister(99))

    explicit = BayesDRModel(Y, T, X)
    fit!(MersenneTwister(99), explicit, BayesDRMCMC(); fit_kwargs...)
    @test explicit.result.posterior_effects == reference.result.posterior_effects
    @test explicit.result.bootstrap_estimates == reference.result.bootstrap_estimates

    defaulted = BayesDRModel(Y, T, X)
    fit!(MersenneTwister(99), defaulted; fit_kwargs...)
    @test defaulted.result.posterior_effects == reference.result.posterior_effects
    @test defaulted.result.bootstrap_estimates == reference.result.bootstrap_estimates
end

@testset "BayesDR p greater than n smoke test" begin
    rng = MersenneTwister(17)
    n, p = 24, 30
    X = randn(rng, n, p)
    T = repeat([0.0, 1.0], n ÷ 2)
    Y = 0.5 .* T .+ X[:, 1] .+ randn(rng, n)
    model = BayesDRModel(Y, T, X)

    fit!(
        model, BayesDRMCMC();
        n_samples = 4, n_burn = 2, n_chains = 1, n_boot = 3, rng,
    )
    @test isfinite(model.result.estimate)
    @test isfinite(model.result.standard_error)
    @test ncovariates(model) > nobs(model)
end
