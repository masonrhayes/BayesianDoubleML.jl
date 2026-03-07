# Real Data VI Tests
# Tests for VI inference on real DoubleML datasets
# PLR and IRM data with both basic and hierarchical models

using BayesianDoubleML
using Test
using Random
using Statistics
using CSV
using DataFrames

@testset "Real Data VI - PLR Dataset (jl2) Hierarchical" begin
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

    println("\n=== Real Data VI: PLR (jl2) Hierarchical ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Observations: $n, Controls: $p")
    println("True α: 0.5")

    Random.seed!(1100)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :hier
    @test length(result.alpha_samples) >= 500

    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("\n=== Results ===")
    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Std α: $(round(alpha_std, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Converged: $(result.converged)")
    println("  Coverage (0.5 in CI): $(ci[1] < 0.5 < ci[2])")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
    @test alpha_std > 0
    @test ci[1] < alpha_mean < ci[2]
    @test isfinite(result.final_elbo)
end

@testset "Real Data VI - PLR Dataset (jl2) Basic" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    println("\n=== Real Data VI: PLR (jl2) Basic ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Model: Basic")

    Random.seed!(1101)
    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :basic

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - PLR Dataset (jl) Hierarchical" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    println("\n=== Real Data VI: PLR (jl) Hierarchical ===")
    println("Dataset: make_plr_CCDDHNR2018_jl.csv")
    println("Observations: $n, Controls: $p")

    Random.seed!(1102)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - IRM Dataset (jl2) Hierarchical" begin
    data_path = joinpath(@__DIR__, "data", "make_irm_data_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test all(D .∈ Ref([0.0, 1.0]))

    println("\n=== Real Data VI: IRM (jl2) Hierarchical ===")
    println("Dataset: make_irm_data_jl2.csv")
    println("Observations: $n, Controls: $p")
    println("True θ: 0.5")
    println("Treatment is binary: $(all(D .∈ Ref([0.0, 1.0])))")

    Random.seed!(1103)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - IRM Dataset (jl2) Basic" begin
    data_path = joinpath(@__DIR__, "data", "make_irm_data_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: IRM (jl2) Basic ===")
    println("Dataset: make_irm_data_jl2.csv")
    println("Model: Basic")

    Random.seed!(1104)
    problem = BDMLProblem(Y, D, X; model_type = :basic)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult
    @test result.model_type == :basic

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - IRM Dataset (jl) Hierarchical" begin
    data_path = joinpath(@__DIR__, "data", "make_irm_data_jl.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])
    n, p = size(X)

    @test all(D .∈ Ref([0.0, 1.0]))

    println("\n=== Real Data VI: IRM (jl) Hierarchical ===")
    println("Dataset: make_irm_data_jl.csv")
    println("Observations: $n, Controls: $p")

    Random.seed!(1105)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  95% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - SimpleVI" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: SimpleVI ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")
    println("Method: SimpleVI")

    Random.seed!(1106)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = SimpleVI()

    elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  Time: $(round(elapsed, digits = 2))s")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - AD Backend Comparison" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: AD Backend Comparison ===")

    backends = [AutoReverseDiff, AutoMooncake]

    for backend in backends
        Random.seed!(1107)

        # Warmup for Mooncake
        if backend == AutoMooncake
            problem_warmup = BDMLProblem(Y, D, X; model_type = :hier)
            _ = fit(problem_warmup, SimpleVI(; ad_backend = AutoMooncake, n_iterations = 50, n_draws = 100))
            Random.seed!(1107)
        end

        problem = BDMLProblem(Y, D, X; model_type = :hier)
        method = UnifiedVI(; ad_backend = backend, n_iterations = 400, n_draws = 400)

        elapsed = @elapsed result = fit(problem, method; n_iterations = 500, n_draws = 500)
        alpha_mean = mean(result.alpha_samples)

        println("  $(nameof(backend)): α=$(round(alpha_mean, digits = 4)), time=$(round(elapsed, digits = 2))s")

        @test isfinite(alpha_mean)
    end
end

@testset "Real Data VI - Coefficient Table" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: Coefficient Table ===")

    Random.seed!(1108)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 400, n_draws = 400)

    ct = coeftable(result)

    @test ct isa CoefTable

    alpha_idx = findfirst(contains("alpha"), ct.rownms)
    @test alpha_idx !== nothing

    alpha_estimate = ct.mat[alpha_idx, 1]

    println("  Coefficient table: $(length(ct.rownms)) parameters")
    println("  Alpha estimate: $(round(alpha_estimate, digits = 4))")
end

@testset "Real Data VI - Convergence and ELBO" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: Convergence and ELBO ===")

    Random.seed!(1109)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    method = UnifiedVI()
    result = fit(problem, method; n_iterations = 600, n_draws = 500)

    @test hasfield(typeof(result), :final_elbo)
    @test hasfield(typeof(result), :converged)
    @test hasfield(typeof(result), :n_iterations)

    @test isfinite(result.final_elbo)
    @test result.n_iterations > 0

    println("  Final ELBO: $(round(result.final_elbo, digits = 2))")
    println("  Converged: $(result.converged)")
    println("  Iterations: $(result.n_iterations)")
end

@testset "Real Data VI - Legacy API" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: Unified VI API ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")

    Random.seed!(1110)
    problem = BDMLProblem(Y, D, X; model_type = :hier)
    result = fit(
        problem, UnifiedVIMethod();
        n_iterations = 500,
        n_draws = 500,
        show_progress = false
    )

    @test result isa BDMLVIResult

    alpha_mean = mean(result.alpha_samples)

    println("  Mean α: $(round(alpha_mean, digits = 4))")
    println("  ELBO: $(round(result.final_elbo, digits = 2))")

    @test isfinite(alpha_mean)
end

@testset "Real Data VI - VI vs MCMC Comparison" begin
    data_path = joinpath(@__DIR__, "data", "make_plr_CCDDHNR2018_jl2.csv")

    @test isfile(data_path)

    df = CSV.read(data_path, DataFrame)
    Y = df.y
    D = df.d
    X_cols = filter(col -> startswith(string(col), "X"), names(df))
    X = Matrix(df[:, X_cols])

    println("\n=== Real Data VI: VI vs MCMC Comparison ===")
    println("Dataset: make_plr_CCDDHNR2018_jl2.csv")

    # VI
    Random.seed!(1111)
    problem_vi = BDMLProblem(Y, D, X; model_type = :hier)
    result_vi = fit(problem_vi, UnifiedVI())
    alpha_vi = mean(result_vi.alpha_samples)
    time_vi = @elapsed fit(problem_vi, UnifiedVI(); n_iterations = 500, n_draws = 500)

    # MCMC
    Random.seed!(1111)
    problem_mcmc = BDMLProblem(Y, D, X; model_type = :hier)
    result_mcmc = fit(problem_mcmc, MCMCNUTS(; n_samples = 300, n_warmup = 150))
    alpha_mcmc = mean(result_mcmc.alpha_samples)

    println("  VI: α=$(round(alpha_vi, digits = 4)), time=$(round(time_vi, digits = 2))s")
    println("  MCMC: α=$(round(alpha_mcmc, digits = 4))")
    println("  Difference: $(round(abs(alpha_vi - alpha_mcmc), digits = 4))")

    # Both should be reasonably close
    @test abs(alpha_vi - alpha_mcmc) < 0.5
end

println("\n=== All Real Data VI Tests Complete ===")
println("Tested: PLR and IRM, both sizes, basic/hierarchical, AD backends, SimpleVI")
