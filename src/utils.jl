function _standardize_vector(x)
    μ = mean(x)
    σ = stdm(x, μ)
    output = Vector{Float64}(undef, length(x))
    scale = 1.0 / Float64(σ)

    @inbounds @simd for i in eachindex(x)
        output[i] = (Float64(x[i]) - Float64(μ)) * scale
    end

    return output, Float64(μ), Float64(σ)
end

function _standardize_column!(output, means, sds, j, column)
    μ = mean(column)
    σ = stdm(column, μ)
    means[j] = Float64(μ)
    sds[j] = Float64(σ)
    scale = 1.0 / sds[j]

    return @inbounds @simd for i in axes(output, 1)
        output[i, j] = (Float64(column[i]) - means[j]) * scale
    end
end

function _standardize_columns!(output, means, sds, columns)
    for (j, column) in enumerate(columns)
        _standardize_column!(output, means, sds, j, column)
    end

    return output
end

function _standardize_data(Y, D, columns, n::Int, p::Int)
    Y_s, Y_mean, Y_sd = _standardize_vector(Y)
    D_s, D_mean, D_sd = _standardize_vector(D)
    X_s = Matrix{Float64}(undef, n, p)
    X_mean = Vector{Float64}(undef, p)
    X_sd = Vector{Float64}(undef, p)
    _standardize_columns!(X_s, X_mean, X_sd, columns)

    stats = StandardizationStats(Y_mean, Y_sd, D_mean, D_sd, X_mean, X_sd)
    return Y_s, D_s, X_s, stats
end

function standardize_data(Y, D, X)
    n = length(Y)
    return _standardize_data(Y, D, eachcol(X), n, size(X, 2))
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
    ad_backend_kwargs(ad_type, ad_kwargs::NamedTuple = (;))

Return AD backend constructor kwargs with backend-specific defaults merged in.
Explicitly provided `ad_kwargs` always take precedence over the defaults.

# Backend-specific defaults
- `AutoReverseDiff`: `compile=false` (required by AdvancedVI >= 0.7, which rejects
  compiled tapes because they freeze captured values at preparation time)
- `AutoForwardDiff`: no kwargs needed; `chunksize=nothing` (the default) already
  means automatic chunk size selection. Do not pass `chunksize=0` — it is invalid.
"""
function ad_backend_kwargs(ad_type, ad_kwargs::NamedTuple = (;))
    if ad_type == AutoReverseDiff && !haskey(ad_kwargs, :compile)
        # AdvancedVI >= 0.7 rejects compiled ReverseDiff tapes (stale gradients)
        ad_kwargs = merge(ad_kwargs, (compile = false,))
    end

    return ad_kwargs
end

"""
    configure_ad_backend(ad_type, ad_kwargs, use_subsample)

Configure AD backend-specific settings (see [`ad_backend_kwargs`](@ref)) and emit
backend-specific usage notices.

# Primary Backends
- AutoReverseDiff: Set compile=false by default (required by AdvancedVI >= 0.7)
- AutoMooncake: No special configuration needed (warn about warmup)

# Secondary Backends
- AutoZygote: Standard configuration
- AutoForwardDiff: Standard configuration (automatic chunk size by default)
"""
function configure_ad_backend(ad_type, ad_kwargs, use_subsample)
    ad_kwargs = ad_backend_kwargs(ad_type, ad_kwargs)

    if ad_type == AutoMooncake
        # Mooncake is fast after warmup, warn about first run
        @info "Using AutoMooncake. First run(s) compile differentiation rules."
        @info "For optimal performance, run warmup iterations or use in batch processing."
    end

    return ad_kwargs
end
