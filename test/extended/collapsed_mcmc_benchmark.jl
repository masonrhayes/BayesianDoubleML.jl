# Standalone experiment (not included by test/runtests.jl or test/extended/runtests.jl).
# Run: julia -t 2 --project=test test/extended/collapsed_mcmc_benchmark.jl
# Optional: BDML_BENCH_SAMPLES=1000 BDML_BENCH_CHAINS=4 julia -t 4 --project=test test/extended/collapsed_mcmc_benchmark.jl

using BayesianDoubleML
using LinearAlgebra
using Printf
using Random
using StableRNGs
using Statistics
using Test

const FCH = BayesianDoubleML.FlexiChains

# Small matrix products are faster without BLAS oversubscription alongside MCMCThreads.
BLAS.set_num_threads(1)

function alpha_diagnostics(result)
    info = chain_info(result)
    draws = reshape(result.alpha_samples, info.n_samples_per_chain, info.n_chains)
    return (
        mean = mean(draws),
        sd = std(draws),
        interval = quantile(vec(draws), [0.025, 0.975]),
        ess = FCH.ess(draws),
        mcse = FCH.mcse(draws),
        rhat = FCH.rhat(draws),
    )
end

function fit_timed(df, kind, method, seed, n_samples, n_chains)
    model = BDMLModel(df, :y, :d; model_type = kind)
    Random.seed!(seed)
    seconds = @elapsed fit!(model, method; n_samples, n_chains)
    return model.result, seconds
end

function compare(df, kind, n_samples, n_chains)
    full_method = MCMCNUTS(; target_acceptance = 0.8, max_depth = 10)
    collapsed_method = CollapsedMCMC(; target_acceptance = 0.8, max_depth = 10)
    seed = 2026

    # Compile both fits outside the measured interval, on the same data and chain layout.
    fit_timed(df, kind, full_method, seed, 20, n_chains)
    fit_timed(df, kind, collapsed_method, seed, 20, n_chains)

    full_result, full_time = fit_timed(df, kind, full_method, seed, n_samples, n_chains)
    collapsed_result, collapsed_time = fit_timed(df, kind, collapsed_method, seed, n_samples, n_chains)
    # Collapsed fits are only milliseconds here; use a median to reduce timer noise.
    collapsed_times = [collapsed_time]
    for _ in 1:9
        _, seconds = fit_timed(df, kind, collapsed_method, seed, n_samples, n_chains)
        push!(collapsed_times, seconds)
    end
    collapsed_time = median(collapsed_times)
    full = alpha_diagnostics(full_result)
    collapsed = alpha_diagnostics(collapsed_result)

    println("\n$kind: n=200, p=100, σ_ε=2, true α=2; $n_chains chains × $n_samples draws")
    println("                   fit (s)  α mean    α sd     95% equal-tailed CI    α ESS    α R̂   α ESS/s")
    for (name, d, seconds) in (("Full NUTS", full, full_time), ("Collapsed NUTS", collapsed, collapsed_time))
        @printf(
            "  %-15s %7.2f  %7.3f  %7.3f   [%6.3f, %6.3f]   %7.1f   %5.3f   %7.1f\n",
            name, seconds, d.mean, d.sd, d.interval..., d.ess, d.rhat, d.ess / seconds
        )
    end
    @printf(
        "  Speedup (total fit): %.2fx; speedup (α ESS/s): %.2fx\n",
        full_time / collapsed_time, (collapsed.ess / collapsed_time) / (full.ess / full_time)
    )
    println("  Fit time excludes compilation and BDMLModel construction; includes SVD. Full: 1 run; collapsed: median of 10.")

    pooled_sd = hypot(full.sd, collapsed.sd) / sqrt(2)
    pooled_mcse = hypot(full.mcse, collapsed.mcse)
    @testset "$kind: posterior agreement and fit speed" begin
        @test all(isfinite, (full.mean, full.sd, full.ess, full.mcse, full.rhat))
        @test all(isfinite, (collapsed.mean, collapsed.sd, collapsed.ess, collapsed.mcse, collapsed.rhat))
        @test abs(full.mean - collapsed.mean) < max(4 * pooled_mcse, 0.25 * pooled_sd)
        @test 0.7 < collapsed.sd / full.sd < 1.3
        @test maximum(abs.(full.interval .- collapsed.interval)) < max(0.15, pooled_sd)
        # This is an opt-in experiment: unlike CI tests, it intentionally checks measured speed.
        @test collapsed_time < full_time
        @test collapsed.ess / collapsed_time > full.ess / full_time
    end
    if min(full.ess, collapsed.ess) < 100 || max(full.rhat, collapsed.rhat) > 1.05
        @warn "Low α ESS or high α R̂: posterior comparison is inconclusive; increase BDML_BENCH_SAMPLES."
    end
    return nothing
end

n_samples = parse(Int, get(ENV, "BDML_BENCH_SAMPLES", "500"))
n_chains = parse(Int, get(ENV, "BDML_BENCH_CHAINS", "2"))
n_samples >= 100 || error("BDML_BENCH_SAMPLES must be at least 100")
n_chains >= 2 || error("BDML_BENCH_CHAINS must be at least 2 for R̂")

# One fixed replication of the paper's Table 1 design, shared across both model
# variants and both MCMC implementations. No cross-method chain identity is implied.
df = make_plr_DTL2025(200, 100, 2.0; alpha = 2.0, rng = StableRNG(42))
println("Julia $(VERSION), $(Threads.nthreads()) Julia threads, $(BLAS.get_num_threads()) BLAS thread(s)")
for kind in (:basic, :hier)
    compare(df, kind, n_samples, n_chains)
end
