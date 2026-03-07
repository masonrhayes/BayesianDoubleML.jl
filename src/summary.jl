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

    # Diagnostics (MCMC vs VI specific)
    if result isa BDMLResult
        print_section_header(io, COLOR_MAGENTA, "MCMC Diagnostics")
        print_mcmc_diagnostics(io, result)
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
    println(io, COLOR_BOLD, "╔══════════════════════════════════════════════════════════════════════╗")
    println(io, "║         Bayesian Double ML Model Summary                   ║")
    return println(io, "╚══════════════════════════════════════════════════════════════════════╝", COLOR_RESET)
end

function print_section_header(io::IO, color::String, title::String)
    line = "─"^(length(title) + 2)
    println(io, color, COLOR_BOLD, title, COLOR_RESET)
    return println(io, color, line, COLOR_RESET)
end

function print_model_info(io::IO, result::AbstractBDMLResult)
    stats = result.std_stats
    @printf io "  Model Type:       %s\n" result.model_type
    return @printf io "  Standardization:  Y mean=%.3f, sd=%.3f; D mean=%.3f, sd=%.3f\n" stats.Y_mean stats.Y_sd stats.D_mean stats.D_sd
end

function print_method_info(io::IO, result::BDMLResult)
    @printf io "  Method:           %s\n" "NUTS (No-U-Turn Sampler)"
    return @printf io "  Samples:          %d\n" length(result.alpha_samples)
end

function print_method_info(io::IO, result::BDMLVIResult)
    # Determine VI type from variational family field
    vi_type = result.variational_family == :fullrank ? "Full-Rank Gaussian" :
        result.variational_family == :lowrank ? "Low-Rank Gaussian" : "Mean-Field Gaussian"
    @printf io "  Method:           Unified VI (%s)\n" vi_type
    @printf io "  Iterations:       %d\n" result.n_iterations
    return @printf io "  Samples Drawn:    %d\n" length(result.alpha_samples)
end

function print_causal_effect(io::IO, result::AbstractBDMLResult)
    alpha_mean = mean(result.alpha_samples)
    alpha_std = std(result.alpha_samples)
    ci = credible_interval(result.alpha_samples)
    @printf io "  Estimate:         %s%.4f%s\n" COLOR_BOLD alpha_mean COLOR_RESET
    @printf io "  Std Error:        %.4f\n" alpha_std
    return @printf io "  95%% CI:           [%s%.4f%s, %s%.4f%s]\n" COLOR_BOLD ci[1] COLOR_RESET COLOR_BOLD ci[2] COLOR_RESET
end

function print_mcmc_diagnostics(io::IO, result::BDMLResult)
    # Try to extract ESS and R-hat from the chain if available
    return try
        ess_val = ess(result)
        @printf io "  ESS (Total):      %.0f\n" ess_val
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
    return if result isa BDMLVIResult
        if result.converged
            println(io, COLOR_GREEN, "✓ Diagnostics: ELBO converged.", COLOR_RESET)
        else
            println(io, COLOR_YELLOW, "⚠ Diagnostics: ELBO may not have converged.", COLOR_RESET)
        end
    else
        # MCMC - simplified
        println(io, COLOR_GREEN, "✓ MCMC sampling complete - review ESS and diagnostics above", COLOR_RESET)
    end
end

# Export summary function
export summary
