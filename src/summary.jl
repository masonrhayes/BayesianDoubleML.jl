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
result = fit(problem, UnifiedVIMethod())
summary(result)

# Or capture output
output = summary(result)
println(output)
```

See also: [`coeftable`](@ref) for tabular coefficient output
"""
function Base.summary(result::AbstractBDMLResult)
    io = IOBuffer()

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
            print_elbo_plot(io, result.diagnostic_history)
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

    output = String(take!(io))
    print(output)
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
    # Determine VI method and family
    if result.vi_method == :simple
        # Simple VI only supports Mean-Field Gaussian
        @printf io "  Method:           Simple VI (Mean-Field Gaussian)\n"
    elseif result.vi_method == :vmp
        @printf io "  Method:           VMP (Conjugate Inverse-Wishart)\n"
    else
        # Unified VI supports multiple families
        vi_type = result.variational_family == :fullrank ? "Full-Rank Gaussian" :
            result.variational_family == :lowrank ? "Low-Rank Gaussian" : "Mean-Field Gaussian"
        @printf io "  Method:           Unified VI (%s)\n" vi_type
    end
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

function print_vmp_diagnostics(io::IO, result::BDMLVMPResult)
    kind_str = result.diagnostic_kind == :elbo ? "ELBO" : "Parameter Change"
    @printf io "  Final Diagnostic: %.2f (%s)\n" result.final_diagnostic kind_str
    @printf io "  Converged:        %s\n" (result.converged ? "$(COLOR_GREEN)Yes ✓$(COLOR_RESET)" : "$(COLOR_RED)No ✗$(COLOR_RESET)")
    return if !isempty(result.diagnostic_history)
        n_iters = length(result.diagnostic_history)
        diag_start = result.diagnostic_history[1]
        diag_end = result.diagnostic_history[end]
        improvement = diag_end - diag_start
        @printf io "  Diagnostic Improvement: %.2f (%.1f%%)\n" improvement (100 * improvement / abs(diag_start))
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
    print_elbo_plot(io::IO, elbo_history::Vector{Float64}; max_points=100)

Print an ASCII line plot of ELBO convergence history.

For long histories (>100 points), intelligently samples to show key features:
- Always shows first point
- Shows every 10th point for middle section
- Shows last 100 points in detail

# Arguments
- `io::IO`: Output stream
- `elbo_history::Vector{Float64}`: Vector of ELBO values
- `max_points::Int=100`: Maximum points to display (default: 100)
"""
function print_elbo_plot(io::IO, elbo_history::Vector{Float64}; max_points::Int = 100)
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
        title = "ELBO Convergence",
        xlabel = "Iteration",
        ylabel = "ELBO",
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
    summary(model::AbstractBDMLModel)

Display a comprehensive summary of a fitted BDML model.

Delegates to the stored result. Throws an error if the model has not been fitted.

# Examples
```julia
model = BDMLModel(Y, D, X; model_type=:hier)
fit!(model)
summary(model)
```
"""
function Base.summary(model::AbstractBDMLModel)
    model.is_fitted || error("Model has not been fitted. Call fit!() first.")
    return Base.summary(model.result)
end
