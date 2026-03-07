# Log-Posterior Computation for Unified VI
# Applies bijector transformation before computing likelihood

using LogDensityProblems
using Distributions
using LinearAlgebra
using Bijectors

export logdensity, compute_log_likelihood

"""
    LogDensityProblems.logdensity(model::BDMLVIModel, θ_unconstrained)

Compute log-posterior for BDML model with bijector transformation.

This is the core function for VI optimization. It:
1. Receives parameters in UNCONSTRAINED space (ℝ^d) from the optimizer
2. Applies bijector transformation to CONSTRAINED space (positive variances, ρ in [0,1])
3. Computes log-prior + log-likelihood in constrained space

# Arguments
- `model::BDMLVIModel`: The model instance
- `θ_unconstrained::Vector{Float64}`: Parameters in unconstrained space (real numbers)

# Returns
Scalar log-posterior value: log_prior + (n_data/n) * log_likelihood

# Notes
The bijector transformation is crucial for AD stability:
- Without it: logpdf(Beta(2,2), ρ_raw) fails when ρ_raw < 0 or ρ_raw > 1
- With it: ρ_raw = logistic(logit_ρ_raw) always in [0,1], logpdf always defined

# Examples
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic)
θ_unconstrained = zeros(LogDensityProblems.dimension(model))
logdens = LogDensityProblems.logdensity(model, θ_unconstrained)
```

# AD Backend Compatibility
This implementation is safe for all AD backends:
- ReverseDiff (primary): Tape-based reverse mode
- Mooncake (primary): Rule-based reverse mode
- Zygote (secondary): Source-to-source reverse mode
- ForwardDiff (secondary): Dual number forward mode

No try/catch, no closures, type-stable throughout.
"""
function LogDensityProblems.logdensity(model::BDMLVIModel, θ_unconstrained)
    n = length(model.Y)
    T = model.T

    # STEP 1: Apply bijector to transform from unconstrained → constrained space
    # This is the key insight: VI optimizes in ℝ^d, but model needs valid ranges
    b = bijector(model)
    θ_constrained, logabsdetjac = Bijectors.with_logabsdet_jacobian(b, θ_unconstrained)

    # STEP 2: Extract parameters (now in constrained space)
    if model.model_type == :hier
        θ_Y, θ_D, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ = unpack_parameters(model, θ_constrained)
        p = length(θ_Y)

        # Hierarchical priors on variances
        log_prior = logpdf(InverseGamma(2.0, 2.0), σ²_δ)
        log_prior += logpdf(InverseGamma(2.0, 2.0), σ²_γ)

        # Hierarchical priors on coefficients (σ²_δ, σ²_γ are variance hyperparameters)
        # For zero-mean normal, logpdf(MvNormal(0, Σ), x) = -0.5*(x'Σ⁻¹x + log|Σ| + d*log(2π))
        # Compute manually to avoid type issues with AD
        log_prior += compute_mvnormal_logpdf_zero_mean(θ_Y, σ²_δ)
        log_prior += compute_mvnormal_logpdf_zero_mean(θ_D, σ²_γ)
    else
        θ_Y, θ_D, σ_U, σ_V, ρ_raw = unpack_parameters(model, θ_constrained)
        p = length(θ_Y)

        # Priors on coefficients
        log_prior = compute_mvnormal_logpdf_zero_mean(θ_Y, 25.0)
        log_prior += compute_mvnormal_logpdf_zero_mean(θ_D, 25.0)
    end

    # Common priors for variance and correlation parameters
    # σ_U, σ_V are already positive (constrained space)
    log_prior += logpdf(truncated(Cauchy(0.0, 2.5), 0.1, Inf), σ_U)
    log_prior += logpdf(truncated(Cauchy(0.0, 2.5), 0.1, Inf), σ_V)

    # ρ_raw is already in [0,1] (constrained space via logistic)
    log_prior += logpdf(Beta(2.0, 2.0), ρ_raw)

    # STEP 4: Transform correlation to [-1, 1] for bivariate normal
    ρ = 2 * ρ_raw - 1

    # STEP 5: Compute log-likelihood using pre-allocated temporaries
    log_lik = compute_log_likelihood!(model, θ_Y, θ_D, σ_U, σ_V, ρ)

    # STEP 6: Scale likelihood by n_data/n for subsampling
    # When n < n_data (mini-batch), scale up to get unbiased gradient
    scaled_log_lik = (model.n_data / n) * log_lik

    # STEP 7: Return total log-posterior with Jacobian adjustment
    # The logabsdetjac term accounts for the change of variables from unconstrained to constrained space
    return log_prior + scaled_log_lik + logabsdetjac
end

"""
    compute_log_likelihood!(model::BDMLVIModel, θ_Y, θ_D, σ_U, σ_V, ρ)

Compute log-likelihood using pre-allocated temporaries.

Uses the model's pre-allocated μ_Y_cache and μ_D_cache to avoid allocations
during AD, which significantly improves performance with all backends.

# Arguments
- `model::BDMLVIModel`: Model with pre-allocated temporaries
- `θ_Y::Vector{Float64}`: Outcome coefficients
- `θ_D::Vector{Float64}`: Treatment coefficients
- `σ_U::Float64`: Outcome standard deviation
- `σ_V::Float64`: Treatment standard deviation
- `ρ::Float64`: Correlation in [-1, 1]

# Returns
Scalar log-likelihood value

# Implementation Notes
- Uses mul!() for in-place matrix multiplication
- Pre-allocated temporaries avoid allocations during gradient computation
- Simple loops (no broadcasting) for better AD compatibility
- Type-stable throughout

# Mathematical Details
Computes bivariate normal log-likelihood:
```
log p(Y, D | X, θ) = Σᵢ -0.5 * (2*log(2π) + log|Σ| + [Uᵢ,Vᵢ]' Σ⁻¹ [Uᵢ,Vᵢ])
```
where Uᵢ = Yᵢ - Xᵢ'θ_Y, Vᵢ = Dᵢ - Xᵢ'θ_D, and Σ is the 2×2 covariance matrix.
"""
function compute_log_likelihood!(model::BDMLVIModel, θ_Y, θ_D, σ_U, σ_V, ρ)
    n = length(model.Y)

    # Compute means (allocates new vectors - needed for AD compatibility)
    # Pre-allocated caches are Float64, but AD needs TrackedReal storage
    μ_Y = model.X * θ_Y
    μ_D = model.X * θ_D

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute determinant: |Σ| = σ_U^2 * σ_V^2 - (ρ*σ_U*σ_V)^2
    det_Σ = Σ_11 * Σ_22 - Σ_12^2

    # Compute inverse elements (2x2 matrix inverse)
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    # Compute log-likelihood (loop is faster for AD than broadcasting)
    ll = zero(eltype(σ_U))
    @inbounds for i in 1:n
        # Residuals
        U_i = model.Y[i] - μ_Y[i]
        V_i = model.D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood contribution
        ll += -0.5 * (2 * log(2π) + log(det_Σ) + mahal)
    end

    return ll
end

"""
    compute_log_likelihood(model::BDMLVIModel, θ_Y, θ_D, σ_U, σ_V, ρ)

Non-mutating version of log-likelihood computation.

Allocates new temporaries - useful for testing or when model temporaries
have wrong size (e.g., after subsampling with different batch size).
"""
function compute_log_likelihood(model::BDMLVIModel, θ_Y, θ_D, σ_U, σ_V, ρ)
    n = length(model.Y)

    # Compute means (allocates new vectors)
    μ_Y = model.X * θ_Y
    μ_D = model.X * θ_D

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute determinant and inverse
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    # Compute log-likelihood
    ll = zero(eltype(σ_U))
    @inbounds for i in 1:n
        U_i = model.Y[i] - μ_Y[i]
        V_i = model.D[i] - μ_D[i]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2
        ll += -0.5 * (2 * log(2π) + log(det_Σ) + mahal)
    end

    return ll
end

"""
    compute_mvnormal_logpdf_zero_mean(x, σ²)

Compute log-pdf of zero-mean multivariate normal with isotropic covariance.

log p(x | 0, σ²I) = -0.5 * (x'x/σ² + d*log(σ²) + d*log(2π))

This avoids creating a mean vector which can cause type issues with AD.
"""
function compute_mvnormal_logpdf_zero_mean(x, σ²)
    d = length(x)
    x_sq_norm = sum(x .^ 2)
    return -0.5 * (x_sq_norm / σ² + d * log(σ²) + d * log(2π))
end
