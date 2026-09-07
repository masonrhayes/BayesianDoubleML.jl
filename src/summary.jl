# Summary functionality for BayesianDoubleML.jl
# Provides formatted output with ELBO plots for VI results

using Printf
using UnicodePlots

# ANSI color codes
const COLOR_BLUE = "\033[34m"
const COLOR_GREEN = "\033[32m"
const COLOR_YELLOW = "\033[33m"
const COLOR_MAGENTA = "\033[35m"
const COLOR_CYAN = "\033[36m"
const COLOR_RED = "\033[31m"
const COLOR_RESET = "\033[0m"
const COLOR_BOLD = "\033[1m"

"""
    Base.summary(result::AbstractBDMLResult)

Display a comprehensive summary of a fitted BDML model result.

For MCMC results, shows sampling statistics and convergence diagnostics.
For VI results, shows ELBO convergence and includes an ASCII plot.

# Examples
```julia
result = fit(problem, CollapsedVI())
summary(result)

# Or capture output
output = summary(result)
println(output)
```

See also: [`coeftable`](@ref) for tabular coefficient output
"""
function Base.summary(io::IO, result::AbstractBDMLResult)
    # Title
    print_title_box(io)

    # Model Information
    print_section_header(io, COLOR_BLUE, "Model Information")
    print_model_info(io, result)

    # Inference Method
    print_section_header(io, COLOR_GREEN, "Inference Method")
    print_method_info(io, result)

    # Diagnostics (MCMC vs VI vs VMP specific)
    if result isa BDMLMCMCResult
        print_section_header(io, COLOR_MAGENTA, "MCMC Diagnostics")
        print_mcmc_diagnostics(io, result)
    elseif result isa BDMLVMPResult
        print_section_header(io, COLOR_MAGENTA, "VMP Diagnostics")
        print_vmp_diagnostics(io, result)

        if !isempty(result.diagnostic_history)
            plot_label = result.diagnostic_kind === :negative_bethe_free_energy ? "BFE" : "ELBO"
            print_elbo_plot(io, result.diagnostic_history; label = plot_label)
        end
    elseif result isa BDMLVIResult
        print_section_header(io, COLOR_MAGENTA, "VI Diagnostics")
        print_vi_diagnostics(io, result)

        # ELBO Plot for VI
        if !isempty(result.elbo_history)
            print_elbo_plot(io, result.elbo_history)
        end
    end

    # Causal Effect
    print_section_header(io, COLOR_YELLOW, "Causal Effect (α)")
    print_causal_effect(io, result)

    # Convergence summary
    print_convergence_summary(io, result)

    return nothing
end

function Base.summary(result::AbstractBDMLResult)
    return Base.summary(stdout, result)
end

function Base.summary(io::IO, result::BayesDRResult)
    print_title_box(io)
    print_section_header(io, COLOR_BLUE, "Model Information")
    @printf io "  Model Type:       Bayes-DR\n"
    @printf io "  Estimand:         ATE (binary treatment)\n"
    @printf io "  Outcome Scale:    original\n"

    print_section_header(io, COLOR_GREEN, "Inference Method")
    @printf io "  Method:           Spike-and-slab Gibbs MCMC\n"
    @printf io "  Posterior Draws:  %d\n" length(result.posterior_effects)
    @printf io "  Bootstrap Draws:  %d\n" length(result.bootstrap_estimates)
    print_bayes_dr_diagnostics(io, result)

    print_section_header(io, COLOR_MAGENTA, "Variance Decomposition")
    @printf io "  Naive Variance:   %.6f\n" result.naive_variance
    @printf io "  Posterior Term:   %.6f\n" result.posterior_variance
    @printf io "  Total Variance:   %.6f\n" result.standard_error^2

    print_section_header(io, COLOR_YELLOW, "Causal Effect (ATE)")
    @printf io "  Estimate:         %s%.4f%s\n" COLOR_BOLD result.estimate COLOR_RESET
    @printf io "  Std Error:        %.4f\n" result.standard_error
    @printf io "  %.0f%% CI:          [%s%.4f%s, %s%.4f%s]\n" (100 * result.level) COLOR_BOLD result.confidence_interval[1] COLOR_RESET COLOR_BOLD result.confidence_interval[2] COLOR_RESET
    @printf io "  Propensity Range: [%.4f, %.4f]\n" minimum(result.propensity_mean) maximum(result.propensity_mean)
    @printf io "  Propensities Clipped: %.2f%%\n" (100 * result.propensity_clipped_fraction)
    return nothing
end

Base.summary(result::BayesDRResult) = Base.summary(stdout, result)

function Base.summary(io::IO, result::BayesDRCurveResult; show_curve::Bool = false)
    print_title_box(io)
    print_section_header(io, COLOR_BLUE, "Model Information")
    @printf io "  Model Type:       Bayes-DR\n"
    @printf io "  Estimand:         Exposure-response curve E[Y(t)]\n"
    @printf io "  Treatment Range:  [%.4f, %.4f]\n" minimum(result.treatment_grid) maximum(result.treatment_grid)
    @printf io "  Grid Points:      %d\n" length(result.treatment_grid)
    @printf io "  Curve Degree:     %d\n" result.curve_degree

    print_section_header(io, COLOR_GREEN, "Inference Method")
    @printf io "  Method:           Spike-and-slab Gibbs MCMC\n"
    @printf io "  Posterior Draws:  %d\n" size(result.posterior_curves, 1)
    @printf io "  Bootstrap Draws:  %d\n" size(result.bootstrap_curves, 1)
    @printf io "  Intervals:        pointwise %.0f%% confidence\n" (100 * result.level)
    @printf io "  Density Ratios Clipped: %.2f%%\n" (100 * result.density_ratio_clipped_fraction)
    print_bayes_dr_diagnostics(io, result)

    derivative = average_derivative(result)
    print_section_header(io, COLOR_MAGENTA, "Average Derivative Diagnostic")
    @printf io "  Treatment Interval: [%.4f, %.4f]\n" derivative.treatment_interval[1] derivative.treatment_interval[2]
    @printf io "  Estimate:           %s%.4f%s\n" COLOR_BOLD derivative.estimate COLOR_RESET
    @printf io "  Std Error:          %.4f\n" derivative.standard_error
    @printf io "  %.0f%% CI:            [%s%.4f%s, %s%.4f%s]\n" (100 * derivative.level) COLOR_BOLD derivative.confidence_interval[1] COLOR_RESET COLOR_BOLD derivative.confidence_interval[2] COLOR_RESET

    print_section_header(io, COLOR_YELLOW, "Exposure-Response Curve")
    for location in eachindex(result.treatment_grid)
        @printf io "  t=%8.4f  E[Y(t)]=%9.4f  SE=%8.4f  CI=[%9.4f, %9.4f]\n" result.treatment_grid[location] result.estimate[location] result.standard_error[location] result.confidence_interval[location, 1] result.confidence_interval[location, 2]
    end
    if show_curve
        print_exposure_response_plot(io, result)
    end
    return nothing
end

Base.summary(result::BayesDRCurveResult; show_curve::Bool = false) =
    Base.summary(stdout, result; show_curve = show_curve)

function print_bayes_dr_diagnostics(io::IO, result::Union{BayesDRResult, BayesDRCurveResult})
    info = chain_info(result)
    @printf io "  Chains:           %d\n" info.n_chains
    ess_value = ess(result)
    if ismissing(ess_value)
        @printf io "  Nuisance ESS:     not available\n"
    else
        @printf io "  Nuisance ESS:     %.1f (minimum)\n" ess_value
    end
    rhat_value = rhat(result)
    if ismissing(rhat_value)
        @printf io "  Nuisance R-hat:   not available\n"
    else
        @printf io "  Nuisance R-hat:   %.3f (maximum)\n" rhat_value
    end
    return nothing
end

function print_title_box(io::IO)
    title = "Bayesian Double ML Model Summary"
    box_width = 70
    content_width = box_width - 2  # Exclude borders
    padding = div(content_width - length(title), 2)
    left_pad = padding
    right_pad = content_width - length(title) - left_pad

    println(io, COLOR_BOLD, "╔", "═"^(box_width - 2), "╗")
    println(io, "║", " "^left_pad, title, " "^right_pad, "║")
    return println(io, "╚", "═"^(box_width - 2), "╝", COLOR_RESET)
end

function print_section_header(io::IO, color::String, title::String)
    println(io)  # Add small amount of whitespace before heading
    line = "─"^(length(title) + 2)
    println(io, color, COLOR_BOLD, title, COLOR_RESET)
    return println(io, color, line, COLOR_RESET)
end

function print_model_info(io::IO, result::AbstractBDMLResult)
    stats = result.std_stats
    @printf io "  Model Type:       %s\n" result.model_type
    return @printf io "  Standardization:  Y mean=%.3f, sd=%.3f; D mean=%.3f, sd=%.3f\n" stats.Y_mean stats.Y_sd stats.D_mean stats.D_sd
end

function print_method_info(io::IO, result::BDMLMCMCResult)
    @printf io "  Method:           %s\n" "NUTS (No-U-Turn Sampler)"
    return @printf io "  Samples:          %d\n" length(result.alpha_samples)
end

function print_method_info(io::IO, result::BDMLVMPResult)
    if result.backend == :rxinfer
        @printf io "  Method:           VMP (Conjugate Inverse-Wishart, RxInfer)\n"
    else
        @printf io "  Method:           VMP (Manual Coordinate Ascent)\n"
    end
    @printf io "  Iterations:       %d (requested %d)\n" result.actual_iterations result.n_iterations
    return @printf io "  Samples Drawn:    %d\n" length(result.alpha_samples)
end

function print_method_info(io::IO, result::BDMLVIResult)
    # CollapsedVI supports mean-field and full-rank Gaussians on the collapsed posterior
    vi_type = result.variational_family == :collapsed_fullrank ? "Full-Rank Gaussian" : "Mean-Field Gaussian"
    @printf io "  Method:           Collapsed VI (%s)\n" vi_type
    @printf io "  Iterations:       %d\n" result.n_iterations
    return @printf io "  Samples Drawn:    %d\n" length(result.alpha_samples)
end

function print_causal_effect(io::IO, result::AbstractBDMLResult)
    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)
    @printf io "  Estimate:         %s%.4f%s\n" COLOR_BOLD alpha_mean COLOR_RESET
    @printf io "  Std Error:        %.4f\n" alpha_std
    @printf io "  95%% CI:           [%s%.4f%s, %s%.4f%s]\n" COLOR_BOLD ci[1] COLOR_RESET COLOR_BOLD ci[2] COLOR_RESET

    # Add HPD interval for MCMC results (more appropriate for skewed posteriors)
    return if result isa BDMLMCMCResult
        try
            hpd = hpd_interval(result.alpha_samples)
            @printf io "  95%% HPD:          [%s%.4f%s, %s%.4f%s]\n" COLOR_BOLD hpd[1] COLOR_RESET COLOR_BOLD hpd[2] COLOR_RESET
        catch
            # HPD not available, skip
        end
    end
end

function print_mcmc_diagnostics(io::IO, result::BDMLMCMCResult)
    # Get chain info
    info = chain_info(result)
    @printf io "  Chains:           %d\n" info.n_chains
    @printf io "  Samples/Chain:    %d\n" info.n_samples_per_chain
    @printf io "  Total Samples:    %d\n" info.total_samples

    # ESS
    try
        ess_val = ess(result)
        ess_pct = 100 * ess_val / info.total_samples
        @printf io "  ESS:              %.0f (%.1f%% efficiency)\n" ess_val ess_pct
        if ess_val > 400
            println(io, COLOR_GREEN, "  ✓ Good effective sample size (ESS > 400)", COLOR_RESET)
        elseif ess_val > 200
            println(io, COLOR_YELLOW, "  ⚠ Acceptable ESS (consider more samples)", COLOR_RESET)
        else
            println(io, COLOR_RED, "  ✗ Low ESS (results may be unreliable)", COLOR_RESET)
        end
    catch
        println(io, "  ESS:              Not available")
    end

    # R-hat (most critical convergence diagnostic)
    try
        rhat_val = rhat(result)
        @printf io "  R-hat:            %.3f\n" rhat_val
        if rhat_val < 1.01
            println(io, COLOR_GREEN, "  ✓ Excellent convergence (R-hat < 1.01)", COLOR_RESET)
        elseif rhat_val < 1.05
            println(io, COLOR_YELLOW, "  ⚠ Acceptable convergence (R-hat < 1.05)", COLOR_RESET)
        else
            println(io, COLOR_RED, "  ✗ Poor convergence (R-hat ≥ 1.05)", COLOR_RESET)
        end
    catch
        println(io, "  R-hat:            Not available")
    end

    # MCSE
    return try
        mcse_val = mcse(result)
        @printf io "  MCSE:             %.4f\n" mcse_val
    catch
        println(io, "  MCSE:             Not available")
    end
end

function vmp_diagnostic_label(kind::Symbol)
    kind === :elbo && return "ELBO"
    kind === :negative_bethe_free_energy && return "Negative Bethe Free Energy"
    return "Parameter Change"
end

function print_vmp_diagnostics(io::IO, result::BDMLVMPResult)
    kind_str = vmp_diagnostic_label(result.diagnostic_kind)
    @printf io "  %-21s %.2f\n" "Final $(kind_str):" result.final_diagnostic
    @printf io "  Converged:        %s\n" (result.converged ? "$(COLOR_GREEN)Yes ✓$(COLOR_RESET)" : "$(COLOR_RED)No ✗$(COLOR_RESET)")
    return if !isempty(result.diagnostic_history)
        n_iters = length(result.diagnostic_history)
        diag_start = result.diagnostic_history[1]
        diag_end = result.diagnostic_history[end]
        improvement = diag_end - diag_start
        @printf io "  %-21s %.2f (%.1f%%)\n" "$(kind_str) Improvement:" improvement (100 * improvement / abs(diag_start))
    end
end

function print_vi_diagnostics(io::IO, result::BDMLVIResult)
    @printf io "  Final ELBO:       %.2f\n" result.final_elbo
    @printf io "  Converged:        %s\n" (result.converged ? "$(COLOR_GREEN)Yes ✓$(COLOR_RESET)" : "$(COLOR_RED)No ✗$(COLOR_RESET)")
    return if !isempty(result.elbo_history)
        n_iters = length(result.elbo_history)
        elbo_start = result.elbo_history[1]
        elbo_end = result.elbo_history[end]
        improvement = elbo_end - elbo_start
        @printf io "  ELBO Improvement: %.2f (%.1f%%)\n" improvement (100 * improvement / abs(elbo_start))
    end
end

"""
    print_elbo_plot(io::IO, elbo_history::Vector{Float64}; label="ELBO", max_points=100)

Print an ASCII line plot of a variational diagnostic convergence history.

For long histories (>100 points), intelligently samples to show key features:
- Always shows first point
- Shows every 10th point for middle section
- Shows last 100 points in detail

# Arguments
- `io::IO`: Output stream
- `elbo_history::Vector{Float64}`: Vector of ELBO values
- `label::AbstractString="ELBO"`: Diagnostic label used in the plot
- `max_points::Int=100`: Maximum points to display (default: 100)
"""
function print_elbo_plot(
        io::IO,
        elbo_history::Vector{Float64};
        label::AbstractString = "ELBO",
        max_points::Int = 100,
    )
    n = length(elbo_history)

    if n == 0
        println(io, "  (No ELBO history available)")
        return
    end

    # Intelligent downsampling for long histories
    if n <= max_points
        # Use all points
        elbo_plot = elbo_history
        x_vals = 1:n
    else
        # Smart sampling: first point + every 10th + last 100
        if n > 1000
            # Very long: sample every 10th, then last 100
            sampled_indices = vcat(1:10:(n - 100), (n - 99):n)
        else
            # Moderate: just last 100
            sampled_indices = (n - 99):n
        end
        elbo_plot = elbo_history[sampled_indices]
        x_vals = sampled_indices
    end

    # Create UnicodePlots line plot
    plot = lineplot(
        x_vals,
        elbo_plot,
        title = "$(label) Convergence",
        xlabel = "Iteration",
        ylabel = label,
        width = 50,
        height = 8,
        border = :ascii
    )

    println(io, plot)

    # Add note if downsampled
    return if n > max_points
        println(io, COLOR_CYAN, "  (Showing ", length(elbo_plot), " of ", n, " iterations)", COLOR_RESET)
    end
end

"""
    print_exposure_response_plot(io::IO, result::BayesDRCurveResult; height=10, width=60)

Print a UnicodePlots line plot of the posterior-mean exposure-response curve
`E[Y(t)]` over the treatment grid, with the pointwise confidence limits drawn
as flanking lines. The grid and intervals come from [`exposure_response_curve`](@ref).

# Arguments
- `io::IO`: Output stream
- `result::BayesDRCurveResult`: Fitted continuous-treatment BayesDR result
- `height::Int=10`: Plot height in characters
- `width::Int=60`: Plot width in characters
"""
function print_exposure_response_plot(
        io::IO,
        result::BayesDRCurveResult;
        height::Int = 10,
        width::Int = 60,
    )
    curve = exposure_response_curve(result)
    order = sortperm(curve.treatment)
    treatment = curve.treatment[order]

    plot = lineplot(
        treatment,
        curve.estimate[order],
        title = "Exposure-Response Curve E[Y(t)]",
        xlabel = "Treatment",
        ylabel = "E[Y(t)]",
        width = width,
        height = height,
        border = :ascii,
        name = "estimate",
        color = :green
    )
    lineplot!(plot, treatment, curve.lower[order]; name = "lower CI", color = :blue)
    lineplot!(plot, treatment, curve.upper[order]; name = "upper CI", color = :red)

    return println(io, plot)
end

function print_convergence_summary(io::IO, result::AbstractBDMLResult)
    println(io, COLOR_BOLD, "─"^62, COLOR_RESET)
    if result isa BDMLVMPResult
        if result.converged
            println(io, COLOR_GREEN, "✓ Diagnostics: VMP converged.", COLOR_RESET)
        else
            println(io, COLOR_YELLOW, "⚠ Diagnostics: VMP may not have converged.", COLOR_RESET)
        end
    elseif result isa BDMLVIResult
        if result.converged
            println(io, COLOR_GREEN, "✓ Diagnostics: ELBO converged.", COLOR_RESET)
        else
            println(io, COLOR_YELLOW, "⚠ Diagnostics: ELBO may not have converged.", COLOR_RESET)
        end
    end
    return nothing
end

# Export summary function
export summary

# Model delegation - allow summary() to be called directly on fitted models

"""
    summary(model::AbstractBDMLModel; show_curve=false)

Display a comprehensive summary of a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.

# Keyword Arguments
- `show_curve::Bool=false`: For continuous-treatment `BayesDRModel`s, also
  print a UnicodePlots rendering of the exposure-response curve. Ignored for
  all other model types.

# Examples
```julia
model = BDMLModel(Y, D, X; model_type=:hier)
fit!(model)
summary(model)
```
"""
function Base.summary(io::IO, model::AbstractBDMLModel; show_curve::Bool = false)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    if show_curve && model.result isa BayesDRCurveResult
        return Base.summary(io, model.result; show_curve = true)
    end
    return Base.summary(io, model.result)
end

function Base.summary(model::AbstractBDMLModel; show_curve::Bool = false)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    return Base.summary(stdout, model; show_curve = show_curve)
end
