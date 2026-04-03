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

Transforms parameters from unconstrained space (ℝ^d) to constrained space
(positive variances, ρ ∈ [0,1]), then computes log-prior + log-likelihood.

# Arguments
- `model::BDMLVIModel`: The model instance
- `θ_unconstrained::Vector{Float64}`: Parameters in unconstrained space

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

    # Transform from unconstrained to constrained space via bijector
    b = bijector(model)
    θ_constrained, logabsdetjac = Bijectors.with_logabsdet_jacobian(b, θ_unconstrained)

    # Compute log-prior
    if model.model_type == :hier
        δ, γ, σ_U, σ_V, ρ_raw, σ²_δ, σ²_γ = unpack_parameters(model, θ_constrained)
        p = length(δ)

        # Hierarchical priors on variances
        log_prior = logpdf(InverseGamma(2.0, 2.0), σ²_δ)
        log_prior += logpdf(InverseGamma(2.0, 2.0), σ²_γ)

        # Hierarchical priors on coefficients (zero-mean MvNormal with variance hyperparameters)
        log_prior += compute_mvnormal_logpdf_zero_mean(δ, σ²_δ)
        log_prior += compute_mvnormal_logpdf_zero_mean(γ, σ²_γ)
    else
        δ, γ, σ_U, σ_V, ρ_raw = unpack_parameters(model, θ_constrained)
        p = length(δ)

        # Priors on coefficients (Equation 12, 5) - fixed variance
        log_prior = compute_mvnormal_logpdf_zero_mean(δ, 25.0)
        log_prior += compute_mvnormal_logpdf_zero_mean(γ, 25.0)
    end

    # Priors for variance and correlation parameters
    log_prior += logpdf(truncated(Cauchy(0.0, 2.5), 0.1, Inf), σ_U)
    log_prior += logpdf(truncated(Cauchy(0.0, 2.5), 0.1, Inf), σ_V)
    log_prior += logpdf(Beta(2.0, 2.0), ρ_raw)

    # Transform correlation to [-1, 1] for bivariate normal
    ρ = 2 * ρ_raw - 1

    # Compute log-likelihood using pre-allocated temporaries
    log_lik = compute_log_likelihood!(model, δ, γ, σ_U, σ_V, ρ)

    # Scale likelihood by n_data/n for subsampling (mini-batch gradient correction)
    scaled_log_lik = (model.n_data / n) * log_lik

    # Total log-posterior with Jacobian adjustment for change of variables
    return log_prior + scaled_log_lik + logabsdetjac
end

"""
    compute_log_likelihood!(model::BDMLVIModel, δ, γ, σ_U, σ_V, ρ)

Compute log-likelihood for the bivariate reduced form model.

# Arguments
- `model::BDMLVIModel`: Model with data
- `δ::Vector{Float64}`: Reduced form coefficients for Y on X ``(Eq. 12)``
- `γ::Vector{Float64}`: Reduced form coefficients for D on X ``(Eq. 5)``
- `σ_U::Float64`: Outcome standard deviation
- `σ_V::Float64`: Treatment standard deviation
- `ρ::Float64`: Correlation in [-1, 1]

# Returns
Scalar log-likelihood value

# Implementation Notes
- Simple loops (no broadcasting) for better AD compatibility
- Type-stable throughout

# Mathematical Details
Computes bivariate normal log-likelihood from Equation 13:
```
``\\log p(Y, D | X, \\delta, \\gamma, \\Sigma) = \\sum_i -0.5 \\cdot (2\\log(2\\pi) + \\log|\\Sigma| + r_i' \\Sigma^{-1} r_i)``
```
where ``r_i = [U_i, V_i] = [Y_i - X_i'\\delta, D_i - X_i'\\gamma]`` and ``\\Sigma`` is the ``2\\times2`` covariance matrix.

See DiTraglia & Liu (2025), Section 4, Equations 13-14.
"""
function compute_log_likelihood!(model::BDMLVIModel, δ, γ, σ_U, σ_V, ρ)
    n = length(model.Y)

    # Reduced form means (Eq. 12, 5)
    μ_Y = model.X * δ
    μ_D = model.X * γ

    # Covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Determinant: |Σ| = σ_U^2 * σ_V^2 - (ρ*σ_U*σ_V)^2
    det_Σ = Σ_11 * Σ_22 - Σ_12^2

    # 2×2 matrix inverse elements
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    # Loop is faster for AD than broadcasting
    ll = zero(eltype(σ_U))
    @inbounds for i in 1:n
        # Reduced form residuals Uᵢ, Vᵢ (Eq. 12, 5)
        U_i = model.Y[i] - μ_Y[i]
        V_i = model.D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' Σ^(-1) [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        ll += -0.5 * (2 * log(2π) + log(det_Σ) + mahal)
    end

    return ll
end

"""
    compute_log_likelihood(model::BDMLVIModel, δ, γ, σ_U, σ_V, ρ)

Non-mutating version of log-likelihood computation.

Allocates new temporaries - useful for testing or when model temporaries
have wrong size (e.g., after subsampling with different batch size).

# Arguments
- `model::BDMLVIModel`: The model instance
- `δ::Vector{Float64}`: Reduced form coefficients for Y on X ``(Eq. 12)``
- `γ::Vector{Float64}`: Reduced form coefficients for D on X ``(Eq. 5)``
- `σ_U::Float64`: Outcome standard deviation
- `σ_V::Float64`: Treatment standard deviation  
- `ρ::Float64`: Correlation in [-1, 1]

# Mathematical Details
Computes bivariate normal log-likelihood from Equation 13:
    ``\\log p(Y, D | X, \\delta, \\gamma, \\Sigma) = \\sum_i -0.5 \\cdot (2\\log(2\\pi) + \\log|\\Sigma| + r_i' \\Sigma^{-1} r_i)``
where ``r_i = [Y_i - X_i'\\delta, D_i - X_i'\\gamma] = [U_i, V_i]`` and ``\\Sigma`` is the ``2\\times2`` covariance matrix.

See DiTraglia & Liu (2025), Section 4, Equations 13-14.
"""
function compute_log_likelihood(model::BDMLVIModel, δ, γ, σ_U, σ_V, ρ)
    n = length(model.Y)

    # Reduced form means (Eq. 12, 5)
    μ_Y = model.X * δ
    μ_D = model.X * γ

    # Covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Determinant and inverse
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = zero(eltype(σ_U))
    @inbounds for i in 1:n
        # Reduced form residuals Uᵢ, Vᵢ (Eq. 12, 5)
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

``\\log p(x | 0, \\sigma^2I) = -0.5 \\cdot (x'x/\\sigma^2 + d\\log(\\sigma^2) + d\\log(2\\pi))``

This avoids creating a mean vector which can cause type issues with AD.
"""
function compute_mvnormal_logpdf_zero_mean(x, σ²)
    d = length(x)
    x_sq_norm = sum(x .^ 2)
    return -0.5 * (x_sq_norm / σ² + d * log(σ²) + d * log(2π))
end
