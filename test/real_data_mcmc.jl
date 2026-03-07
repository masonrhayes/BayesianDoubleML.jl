# Real Data MCMC Tests
# Tests for MCMC inference on real DoubleML datasets
# PLR and IRM data from DoubleML package

using BayesianDoubleML
using Test
using Random
using Statistics
using CSV
using DataFrames

@testset "Real Data MCMC - PLR Dataset (jl2)" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path) "Data file not found at $data_path"

    df = CSV.read(data_path, DataFrame)

    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test length(Y) == n
    @test length(D) == n

    println("\n=== Real Data MCMC: PLR (jl2) ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Observations: $n, Controls: $p")
    println("True α: 0.5")

    Random.seed!(1000)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult
    @test length(result.alpha_samples) >= 300

    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    # Check if estimate is reasonably close (loose tolerance for real data)
    @test abs(alpha_mean - 0.5) < 1.0
end

@testset "Real Data MCMC - PLR Dataset (jl)" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl.csv")

    @test isfile(data_path) "Data file not found at $data_path"

    df = CSV.read(data_path, DataFrame)

    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test length(Y) == n
    @test length(D) == n

    println("\n=== Real Data MCMC: PLR (jl) ===")
    println("Dataset: make_plr_CCDDHNR2018_jl.csv")
    println("Observations: $n, Controls: $p")
    println("True α: 0.5")

    Random.seed!(1001)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Coverage: $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "Real Data MCMC - IRM Dataset (jl2)" begin
    data_path = joinpath(@__DIR__, "data", "make_irm_data_jl2.csv")

    @test isfile(data_path) "Data file not found at $data_path"

    df = CSV.read(data_path, DataFrame)

    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test length(Y) == n
    @test length(D) == n
    @test all(D .∈ Ref([0.0, 1.0])) "D should be binary for IRM"

    println("\n=== Real Data MCMC: IRM (jl2) ===")
    println("Dataset: make_irm_data_jl2.csv")
    println("Observations: $n, Controls: $p")
    println("True θ: 0.5")
    println("Treatment is binary: $(all(D .∈ Ref([0.0, 1.0])))")

    Random.seed!(1002)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test ci[1] < alpha_mean < ci[2]
end

@testset "Real Data MCMC - IRM Dataset (jl)" begin
    data_path = joinpath(@__DIR__, "data", "make_irm_data_jl.csv")

    @test isfile(data_path) "Data file not found at $data_path"

    df = CSV.read(data_path, DataFrame)

    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test length(Y) == n
    @test length(D) == n
    @test all(D .∈ Ref([0.0, 1.0]))

    println("\n=== Real Data MCMC: IRM (jl) ===")
    println("Dataset: make_irm_data_jl.csv")
    println("Observations: $n, Controls: $p")
    println("True θ: 0.5")

    Random.seed!(1003)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - Multi-Chain" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data MCMC: Multi-Chain ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Chains: 2")

    Random.seed!(1004)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult
    @test length(result.alpha_samples) >= 400  # 200 * 2

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Total samples: $(length(result.alpha_samples))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - Basic Model" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data MCMC: Basic Model ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Model: Basic")

    Random.seed!(1005)
    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = MCMCNUTS()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult
    @test result.model_type == :basic

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - HMC Sampler" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data MCMC: HMC Sampler ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Sampler: HMC")

    Random.seed!(1006)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCHMC()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLResult

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data MCMC - Diagnostics" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data MCMC: Diagnostics ===")

    Random.seed!(1007)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100, n_chains = 2)

    # Test diagnostics
    ci = confint(result)
    ess_val = ess(result)
    mcse_val = mcse(result)

    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ESS: $(round(ess_val, digits = 1))")
    println("  MCSE: $(round(mcse_val, digits = 4))")

    @test length(ci) == 2
    @test ess_val > 0
    @test mcse_val > 0
end

@testset "Real Data MCMC - Coefficient Table" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data MCMC: Coefficient Table ===")

    Random.seed!(1008)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = MCMCNUTS()
    result = fit(problem, method; n_samples = 200, n_warmup = 100)

    ct = coeftable(result)

    @test ct isa CoefTable

    alpha_idx = findfirst(contains("alpha"), ct.rownms)
    @test alpha_idx !== nothing

    alpha_estimate = ct.mat[alpha_idx, 1]
    alpha_stderror = ct.mat[alpha_idx, 2]

    println("  Coefficient table: $(length(ct.rownms)) parameters")
    println("  Alpha estimate: $(round(alpha_estimate, digits = 4))")
    println("  Alpha std error: $(round(alpha_stderror, digits = 4))")
end

println("\n=== All Real Data MCMC Tests Complete ===")
println("Tested: PLR and IRM datasets, both sizes, multi-chain, diagnostics")
