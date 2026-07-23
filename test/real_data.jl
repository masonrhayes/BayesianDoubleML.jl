# Real Data Tests (Consolidated)
# Tests for MCMC and VI inference on actual DoubleML datasets with new Model API
# Uses only the 2 existing data files in test/data/

using BayesianDoubleML
using Test
using Random
using Statistics
using CSV
using DataFrames
using Mooncake

# ============================================================================
# Shared Data Loading Utility
# ============================================================================

function load_plr_data()
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_n500_p20.csv")

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    return Y, D, X, data_path
end

function load_irm_data()
    data_path = joinpath(@__DIR__, "data", "make_irm_data_n500_p20.csv")

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    return Y, D, X, data_path
end

# ============================================================================
# MCMC Tests on Real Data
# ============================================================================

@testset "Real Data MCMC - PLR Dataset" begin
    Y, D, X, data_path = load_plr_data()
    n, p = size(X)

    println("\n=== Real Data MCMC: PLR ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Observations: $n, Controls: $p")
    println("True α: 0.5")

    Random.seed!(1000)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed fit!(model, method; n_samples = 500)

    @test isfitted(model)
    @test model.result isa BDMLMCMCResult
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))
    ci = credible_interval(model)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test abs(alpha_mean - 0.5) < 1.0
end

@testset "Real Data MCMC - IRM Dataset" begin
    Y, D, X, data_path = load_irm_data()
    n, p = size(X)

    println("\n=== Real Data MCMC: IRM ===")
    println("Dataset: make_irm_data_n500_p20.csv")
    println("Observations: $n, Controls: $p")
    println("True θ: 0.5")
    println("Treatment is binary: $(all(D .∈ Ref([0.0, 1.0])))")

    Random.seed!(1002)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed fit!(model, method; n_samples = 500)

    @test isfitted(model)
    @test model.result isa BDMLMCMCResult

    alpha_mean = mean(extract_alpha(model))
    ci = credible_interval(model)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "Real Data MCMC - Multi-Chain" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data MCMC: Multi-Chain ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Chains: 2")

    Random.seed!(1004)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed fit!(model, method; n_samples = 300)

    @test isfitted(model)
    @test model.result isa BDMLMCMCResult
    @test length(model.result.alpha_samples) >= 300

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Total samples: $(length(model.result.alpha_samples))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - Basic Model" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data MCMC: Basic Model ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Model: Basic")

    Random.seed!(1005)
    model = BDMLModel(Y, D, X; model_type = :basic)
    method = MCMCNUTS()

    elapsed = @elapsed fit!(model, method; n_samples = 500)

    @test isfitted(model)
    @test model.result isa BDMLMCMCResult
    @test model.result.model_type == :basic

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - Diagnostics" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data MCMC: Diagnostics ===")

    Random.seed!(1007)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    # Test diagnostics via model
    ci = confint(model)
    ess_val = ess(model)
    mcse_val = mcse(model)

    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ESS: $(round(ess_val, digits = 1))")
    println("  MCSE: $(round(mcse_val, digits = 4))")

    @test length(ci) == 2
    @test ess_val > 0
    @test mcse_val > 0
end

@testset "Real Data MCMC - Coefficient Table" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data MCMC: Coefficient Table ===")

    Random.seed!(1008)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    fit!(model, method; n_samples = 200)

    ct = coeftable(model)

    @test ct isa BDMLCoeftable

    alpha_idx = findfirst(contains("α"), ct.coefnames)
    @test alpha_idx !== nothing

    alpha_estimate = ct.coef[alpha_idx]
    alpha_stderror = ct.stderror[alpha_idx]

    println("  Coefficient table: $(length(ct.coefnames)) parameters")
    println("  Alpha estimate: $(round(alpha_estimate, digits = 4))")
    println("  Alpha std error: $(round(alpha_stderror, digits = 4))")
end

# ============================================================================
# VI Tests on Real Data
# ============================================================================

@testset "Real Data VI - PLR Hierarchical" begin
    Y, D, X, data_path = load_plr_data()
    n, p = size(X)

    println("\n=== Real Data VI: PLR Hierarchical ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Observations: $n, Controls: $p")
    println("True α: 0.5")

    Random.seed!(1100)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 200, n_draws = 1000)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :hier
    @test length(model.result.alpha_samples) >= 500

    alpha_mean = mean(extract_alpha(model))
    alpha_std = std(extract_alpha(model))
    ci = credible_interval(model)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Converged: $(model.result.converged)")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test isfinite(model.result.final_elbo)
end

@testset "Real Data VI - PLR Basic" begin
    Y, D, X, data_path = load_plr_data()
    n, p = size(X)

    println("\n=== Real Data VI: PLR Basic ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Model: Basic")

    Random.seed!(1101)
    model = BDMLModel(Y, D, X; model_type = :basic)
    method = UnifiedVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 200, n_draws = 1000)

    @test isfitted(model)
    @test model.result isa BDMLVIResult
    @test model.result.model_type == :basic

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - IRM Hierarchical" begin
    Y, D, X, data_path = load_irm_data()
    n, p = size(X)

    println("\n=== Real Data VI: IRM Hierarchical ===")
    println("Dataset: make_irm_data_n500_p20.csv")
    println("Observations: $n, Controls: $p")
    println("True θ: 0.5")

    Random.seed!(1102)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed fit!(model, method; n_iterations = 200, n_draws = 500)

    @test isfitted(model)
    @test model.result isa BDMLVIResult

    alpha_mean = mean(extract_alpha(model))
    ci = credible_interval(model)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(model.result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "Real Data VI - SimpleVI" begin
    Y, D, X, _ = load_plr_data()
    n, p = size(X)

    println("\n=== Real Data VI: SimpleVI ===")
    println("Dataset: make_plr_CCDDHNR2018_n500_p20.csv")
    println("Observations: $n, Controls: $p")

    Random.seed!(1106)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = SimpleVIMethod()

    elapsed = @elapsed fit!(model, method; n_iterations = 250, n_draws = 1000)

    @test isfitted(model)
    @test model.result isa BDMLVIResult

    alpha_mean = mean(extract_alpha(model))

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test abs(alpha_mean - 0.5) < 1.0
end

@testset "Real Data VI - Coefficient Table" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data VI: Coefficient Table ===")

    Random.seed!(1108)
    model = BDMLModel(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    fit!(model, method; n_iterations = 200, n_draws = 1000)

    ct = coeftable(model)

    @test ct isa BDMLCoeftable

    alpha_idx = findfirst(contains("α"), ct.coefnames)
    @test alpha_idx !== nothing

    alpha_estimate = ct.coef[alpha_idx]

    println("  Coefficient table: $(length(ct.coefnames)) parameters")
    println("  Alpha estimate: $(round(alpha_estimate, digits = 4))")
end

@testset "Real Data VI - MCMC Comparison via Models" begin
    Y, D, X, _ = load_plr_data()

    println("\n=== Real Data VI vs MCMC via Models ===")

    # VI
    model_vi = BDMLModel(Y, D, X; model_type = :hier)
    fit!(model_vi, UnifiedVI(); n_iterations = 250, n_draws = 1000)
    alpha_vi = mean(extract_alpha(model_vi))

    # MCMC
    model_mcmc = BDMLModel(Y, D, X; model_type = :hier)
    fit!(model_mcmc, MCMCNUTS(); n_samples = 250)
    alpha_mcmc = mean(extract_alpha(model_mcmc))

    println("  VI: α = $(round(alpha_vi, digits = 4))")
    println("  MCMC: α = $(round(alpha_mcmc, digits = 4))")
    println("  Difference: $(round(abs(alpha_vi - alpha_mcmc), digits = 4))")

    # Both should be close to 0.5 and to each other
    @test abs(alpha_vi - 0.5) < 1.0
    @test abs(alpha_mcmc - 0.5) < 1.0
    @test abs(alpha_vi - alpha_mcmc) < 0.5
end

println("\n=== All Real Data Tests Complete ===")
println("Tested: MCMC and VI on PLR and IRM datasets with new Model API")
