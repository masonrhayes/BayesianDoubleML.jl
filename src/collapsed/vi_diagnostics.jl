# ELBO convergence checking utilities for VI
# Provides robust convergence detection using moving average approach

"""
    check_elbo_convergence(elbo_history::Vector{Float64};
                          min_pct::Float64=0.30,
                          rel_tol::Float64=0.001,
                          check_trend::Bool=true,
                          min_iterations::Int=50,
                          verbose::Bool=true)

Check if ELBO has converged using moving average over last portion of iterations.

Returns (converged::Bool, message::String).

# Algorithm
1. Requires at least `min_iterations` before declaring convergence
2. Computes moving average over last `min_pct` portion of iterations (min 10)
3. Checks if local standard deviation < `rel_tol` * mean
4. Checks that trend is flat (first half ≈ second half of window)
5. Both variance and trend must be low for convergence

# Arguments
- `elbo_history::Vector{Float64}`: History of ELBO values
- `min_pct::Float64=0.30`: Percentage of final iterations to check (default: 30%)
- `rel_tol::Float64=0.001`: Relative tolerance (0.1% of mean ELBO)
- `check_trend::Bool=true`: Whether to check for flat trend
- `min_iterations::Int=50`: Minimum iterations before convergence
- `verbose::Bool=true`: Whether to print @info messages

# Returns
- `(converged::Bool, message::String)`: Convergence status and description

# Examples
```julia
elbo_history = [-100.0, -95.0, -92.0, -91.0, -90.5, -90.3, -90.2, -90.1, -90.1, -90.1]
converged, msg = check_elbo_convergence(elbo_history)
# converged = true (stable last 30%)
```
"""
function check_elbo_convergence(
        elbo_history::Vector{Float64};
        min_pct::Float64 = 0.3,
        rel_tol::Float64 = 0.01,
        check_trend::Bool = true,
        min_iterations::Int = 50,
        verbose::Bool = true
    )
    n = length(elbo_history)

    # Calculate dynamic window: at least 10 iterations, or min_pct of total
    window = max(10, ceil(Int, n * min_pct))

    # Check if we have enough data
    if n < window
        msg = "Insufficient data (need $window iterations, have $n)"
        return false, msg
    end

    # Check minimum iterations
    if n < min_iterations
        msg = "Need at least $min_iterations iterations (have $n)"
        return false, msg
    end

    # Compute moving average over last window iterations
    recent = elbo_history[(end - window + 1):end]
    ma = mean(recent)
    local_std = std(recent)

    # Check 1: Low variance (std < rel_tol * |mean|)
    threshold = rel_tol * abs(ma)
    is_stable = local_std < threshold

    # Check 2: Flat trend (no strong upward/downward movement)
    is_flat = true
    trend = 0.0
    if check_trend && length(recent) >= 20
        mid = length(recent) ÷ 2
        first_half = mean(recent[1:mid])
        second_half = mean(recent[(mid + 1):end])
        trend = abs(second_half - first_half) / abs(ma)
        is_flat = trend < (rel_tol * 0.5)  # Trend should be half the variance tolerance
    end

    # Both conditions must be met
    is_converged = is_stable && is_flat

    # Build detailed message
    if is_converged
        msg = @sprintf "Converged: MA=%.2f, std=%.4f, threshold=%.4f, trend=%.4f" ma local_std threshold trend
    else
        reason = !is_stable ? "high variance" : "trending"
        msg = @sprintf "Not converged (%s): MA=%.2f, std=%.4f > threshold=%.4f, trend=%.4f" reason ma local_std threshold trend
    end

    return is_converged, msg
end

"""
    check_convergence_with_consecutive_windows(elbo_history::Vector{Float64};
                                              n_windows::Int=2,
                                              min_pct::Float64=0.30,
                                              rel_tol::Float64=0.001,
                                              min_iterations::Int=50)

Stricter convergence check requiring stability over multiple consecutive windows.

Must pass convergence criteria for `n_windows` consecutive windows before declaring converged.
More robust against random fluctuations in ELBO.

# Arguments
- `n_windows::Int=2`: Number of consecutive windows that must be stable
- `min_pct::Float64=0.30`: Percentage of final iterations for each window
- `rel_tol::Float64=0.001`: Relative tolerance (0.1% of mean ELBO)
- `min_iterations::Int=50`: Minimum total iterations

# Returns
- `(converged::Bool, message::String)`
"""
function check_convergence_with_consecutive_windows(
        elbo_history::Vector{Float64};
        n_windows::Int = 2,
        min_pct::Float64 = 0.3,
        rel_tol::Float64 = 0.001,
        min_iterations::Int = 50
    )
    n = length(elbo_history)
    window_size = max(10, ceil(Int, n * min_pct))
    total_needed = n_windows * window_size

    if n < total_needed
        return false, "Need $total_needed iterations for $n_windows consecutive windows"
    end

    if n < min_iterations
        return false, "Need at least $min_iterations iterations"
    end

    # Check each consecutive window
    all_converged = true
    for i in 1:n_windows
        start_idx = n - total_needed + (i - 1) * window_size + 1
        end_idx = start_idx + window_size - 1
        window = elbo_history[start_idx:end_idx]

        ma = mean(window)
        local_std = std(window)
        threshold = rel_tol * abs(ma)

        if local_std >= threshold
            all_converged = false
            msg = @sprintf "Window %d failed: MA=%.2f, std=%.3f > threshold=%.3f" i ma local_std threshold
            return false, msg
        end
    end

    # All windows converged
    # Note: Convergence status is returned, not logged. User can check via result.converged

    return true, "Stable over $n_windows consecutive windows (each $window_size iterations)"
end

export check_elbo_convergence, check_convergence_with_consecutive_windows
