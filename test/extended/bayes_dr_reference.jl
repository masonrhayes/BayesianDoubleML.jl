using BayesianDoubleML
using CSV
using DataFrames
using Printf
using Random
using Test

reference_dir = joinpath(@__DIR__, "..", "reference")
expected = CSV.read(joinpath(reference_dir, "bayes_dr_expected.csv"), DataFrame)

binary_row = only(expected[expected.scenario .== "binary", :])
continuous_rows = expected[expected.scenario .== "continuous", :]

@testset "Binary Bayes-DR R reference" begin
    data = CSV.read(joinpath(reference_dir, "bayes_dr_binary.csv"), DataFrame)
    model = BayesDRModel(data, :y, :treatment)
    fit!(
        model, BayesDRMCMC();
        n_samples = 250,
        n_burn = 500,
        thin = 2,
        n_chains = 2,
        n_boot = 500,
        rng = MersenneTwister(2401),
    )
    global binary_result = model.result
    @test model.result.estimate ≈ binary_row.estimate atol = 0.35
    @test model.result.standard_error ≈ binary_row.standard_error atol = 0.35
    @test model.result.confidence_interval[1] ≈ binary_row.lower atol = 0.6
    @test model.result.confidence_interval[2] ≈ binary_row.upper atol = 0.6
end

@testset "Continuous Bayes-DR R reference" begin
    data = CSV.read(joinpath(reference_dir, "bayes_dr_continuous.csv"), DataFrame)
    grid = collect(continuous_rows.location)
    model = BayesDRModel(data, :y, :treatment)
    fit!(
        model, BayesDRMCMC();
        n_samples = 250,
        n_burn = 500,
        thin = 2,
        n_chains = 2,
        n_boot = 500,
        treatment_grid = grid,
        rng = MersenneTwister(2402),
    )
    global continuous_result = model.result
    @test model.result.estimate ≈ continuous_rows.estimate atol = 0.6
    @test model.result.standard_error ≈ continuous_rows.standard_error atol = 0.6
    @test model.result.confidence_interval[:, 1] ≈ continuous_rows.lower atol = 0.9
    @test model.result.confidence_interval[:, 2] ≈ continuous_rows.upper atol = 0.9
end

format_number(x) = ismissing(x) ? "—" : @sprintf("%.4f", x)
format_number(x::AbstractVector) = join((format_number(v) for v in x), ", ")

function comparison_row(label, reference_value, julia_value, tolerance)
    difference = abs(julia_value - reference_value)
    status = difference <= tolerance ? "pass" : "fail"
    return "| $label | $(format_number(reference_value)) | $(format_number(julia_value)) | " *
        "$(format_number(difference)) | $tolerance | $status |"
end

report = String[]
push!(report, "# Bayes-DR Julia–R Reference Report")
push!(report, "")
push!(
    report, "Comparison of Julia `BayesDRModel` fits against `DoublyRobustHD` " *
        "version `0.0.0.9000`, Git commit `617945098b2f540039367d4398068ba9ae713891`."
)
push!(report, "")
push!(report, "- Julia package: BayesianDoubleML v$(pkgversion(BayesianDoubleML))")
push!(
    report, "- Sampler: `BayesDRMCMC` with 250 retained draws per chain, " *
        "500 burn-in iterations, thinning 2, 2 chains, 500 bootstrap replicates"
)
push!(
    report, "- Fixtures: `bayes_dr_binary.csv` (n = 30, binary treatment) and " *
        "`bayes_dr_continuous.csv` (n = 24, continuous treatment)"
)
push!(
    report, "- Tolerances are broad Monte Carlo bounds; these are " *
        "characterization checks, not exact equivalence tests"
)
push!(report, "")
push!(report, "## Binary treatment")
push!(report, "")
push!(report, "| Quantity | R reference | Julia | Absolute difference | Tolerance | Status |")
push!(report, "|---|---:|---:|---:|---:|---|")
push!(
    report, comparison_row(
        "Estimate", binary_row.estimate, binary_result.estimate, 0.35
    )
)
push!(
    report, comparison_row(
        "Standard error", binary_row.standard_error, binary_result.standard_error, 0.35
    )
)
push!(
    report, comparison_row(
        "CI lower bound", binary_row.lower, binary_result.confidence_interval[1], 0.6
    )
)
push!(
    report, comparison_row(
        "CI upper bound", binary_row.upper, binary_result.confidence_interval[2], 0.6
    )
)
push!(report, "")
push!(report, "## Continuous treatment")
push!(report, "")
push!(report, "| Quantity | Location | R reference | Julia | Absolute difference | Tolerance | Status |")
push!(report, "|---|---:|---:|---:|---:|---:|---|")
for (label, column, julia_values, tolerance) in (
        ("Estimate", :estimate, continuous_result.estimate, 0.6),
        ("Standard error", :standard_error, continuous_result.standard_error, 0.6),
        ("CI lower bound", :lower, continuous_result.confidence_interval[:, 1], 0.9),
        ("CI upper bound", :upper, continuous_result.confidence_interval[:, 2], 0.9),
    )
    for (row, reference_value, julia_value) in
        zip(eachrow(continuous_rows), continuous_rows[:, column], julia_values)
        push!(
            report,
            "| $label | $(format_number(row.location)) | $(format_number(reference_value)) | " *
                "$(format_number(julia_value)) | " *
                "$(format_number(abs(julia_value - reference_value))) | $tolerance | " *
                "$(abs(julia_value - reference_value) <= tolerance ? "pass" : "fail") |"
        )
    end
end
push!(report, "")

report_path = joinpath(reference_dir, "bayes_dr_report.md")
write(report_path, join(report, "\n"))
println("Wrote Bayes-DR reference report to ", report_path)
