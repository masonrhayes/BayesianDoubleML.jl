function standardize_data(Y, D, X)
    Y_mean, Y_sd = mean(Y), std(Y)
    D_mean, D_sd = mean(D), std(D)
    X_mean = mean(X, dims = 1)[:]
    X_sd = std(X, dims = 1)[:]

    Y_s = (Y .- Y_mean) ./ Y_sd
    D_s = (D .- D_mean) ./ D_sd
    X_s = (X .- X_mean') ./ X_sd'

    stats = StandardizationStats(Y_mean, Y_sd, D_mean, D_sd, X_mean, X_sd)

    return Y_s, D_s, X_s, stats
end

function posterior_summary(samples; probs = [0.025, 0.25, 0.5, 0.75, 0.975])
    return Dict(
        :mean => mean(samples),
        :std => std(samples),
        :quantiles => quantile(samples, probs),
        :prob_range => probs
    )
end

function credible_interval(samples; level = 0.95)
    alpha = 1 - level
    lower = alpha / 2
    upper = 1 - alpha / 2
    return quantile(samples, [lower, upper])
end

"""
    configure_ad_backend(ad_type, ad_kwargs, use_subsample)

Configure AD backend-specific settings.

# Primary Backends
- AutoReverseDiff: Set compile=true by default
- AutoMooncake: No special configuration needed (warn about warmup)

# Secondary Backends
- AutoZygote: Standard configuration
- AutoForwardDiff: Use chunk size 0 (auto) by default
"""
function configure_ad_backend(ad_type, ad_kwargs, use_subsample)
    if ad_type == AutoReverseDiff
        # Enable tape compilation for better performance
        if !haskey(ad_kwargs, :compile)
            ad_kwargs = merge(ad_kwargs, (compile = true,))
        end
    elseif ad_type == AutoMooncake
        # Mooncake is fast after warmup, warn about first run
        @info "Using AutoMooncake. First run(s) compile differentiation rules."
        @info "For optimal performance, run warmup iterations or use in batch processing."
    elseif ad_type == AutoForwardDiff
        # Auto chunk size for forward mode
        if !haskey(ad_kwargs, :chunksize)
            ad_kwargs = merge(ad_kwargs, (chunksize = 0,))
        end
    end

    return ad_kwargs
end
