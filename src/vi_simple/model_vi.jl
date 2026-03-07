# VI-Compatible BDML Models
# These models use a direct correlation parameterization (ρ ~ Beta)
# instead of LKJCholesky to ensure compatibility with Turing's ADVI

"""
    bdml_basic_vi(Y, D, X)

Basic BDML model with VI-compatible correlation parameterization.

This version uses a direct ρ ~ Beta(2, 2) prior instead of LKJCholesky,
which is compatible with Turing's ADVI variational inference.

# Model Specification
- θ_Y ~ N(0, 25*I) - Coefficients for outcome equation
- θ_D ~ N(0, 25*I) - Coefficients for treatment equation  
- σ_U ~ Cauchy+(0, 2.5) - Outcome error scale
- σ_V ~ Cauchy+(0, 2.5) - Treatment error scale
- ρ ~ Beta(2, 2) - Correlation (mapped to [-1, 1])

The correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1].

# Notes
- This model is identical to bdml_basic() statistically, but uses a 
  correlation parameterization compatible with ADVI
- For causal inference, α = ρ * σ_U / σ_V is computed post-hoc
"""
@model function bdml_basic_vi(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    θ_Y ~ MvNormal(zeros(T, p), T(25.0) * I)
    θ_D ~ MvNormal(zeros(T, p), T(25.0) * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation: Beta(2, 2) on [0, 1], then map to [-1, 1]
    # Using Beta(2, 2) gives a distribution centered at 0.5 with moderate spread
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1  # Map from [0, 1] to [-1, 1]

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    μ_Y = X * θ_Y
    μ_D = X * θ_D

    # Compute log-likelihood manually using the bivariate normal formula
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = zero(T)
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood for bivariate normal
        ll += -T(0.5) * (2 * log(T(2π)) + log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end

"""
    bdml_hier_vi(Y, D, X)

Hierarchical BDML model with VI-compatible correlation parameterization.

This version uses a direct ρ ~ Beta(2, 2) prior instead of LKJCholesky,
which is compatible with Turing's ADVI variational inference.

# Model Specification
- σ²_δ ~ InvGamma(2, 2) - Variance hyperparameter for θ_Y
- σ²_γ ~ InvGamma(2, 2) - Variance hyperparameter for θ_D
- θ_Y ~ N(0, σ²_δ*I) - Coefficients for outcome equation
- θ_D ~ N(0, σ²_γ*I) - Coefficients for treatment equation  
- σ_U ~ Cauchy+(0, 2.5) - Outcome error scale
- σ_V ~ Cauchy+(0, 2.5) - Treatment error scale
- ρ ~ Beta(2, 2) - Correlation (mapped to [-1, 1])

# Notes
- This model is identical to bdml_hier() statistically, but uses a 
  correlation parameterization compatible with ADVI
- The hierarchical structure provides adaptive shrinkage on coefficients
"""
@model function bdml_hier_vi(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Hierarchical priors
    σ2_δ ~ InverseGamma(T(2), T(2))
    σ2_γ ~ InverseGamma(T(2), T(2))

    θ_Y ~ MvNormal(zeros(T, p), σ2_δ * I)
    θ_D ~ MvNormal(zeros(T, p), σ2_γ * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation: Beta(2, 2) on [0, 1], then map to [-1, 1]
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1  # Map from [0, 1] to [-1, 1]

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    μ_Y = X * θ_Y
    μ_D = X * θ_D

    # Compute log-likelihood manually using the bivariate normal formula
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = zero(T)
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood for bivariate normal
        ll += -T(0.5) * (2 * log(T(2π)) + log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end

"""
    bdml_basic_vi_rd(Y, D, X)

ReverseDiff-optimized version of bdml_basic_vi for large p.

Uses tight @inbounds loops for compact ReverseDiff tape.
Best for: p > 50 with AutoReverseDiff(; compile=true)
"""
@model function bdml_basic_vi_rd(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Priors - explicitly typed for type stability
    θ_Y ~ MvNormal(zeros(T, p), T(25.0) * I)
    θ_D ~ MvNormal(zeros(T, p), T(25.0) * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute means
    μ_Y = X * θ_Y
    μ_D = X * θ_D

    # TIGHT LOOP for ReverseDiff efficiency
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    # Guard against numerical issues with log
    log_det = log(max(det_Σ, eps(T)))
    const_term = -T(0.5) * (2 * log(T(2π)) + log_det)

    ll = zero(T)
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        ll += const_term - T(0.5) * mahal
    end

    Turing.@addlogprob! ll
end

"""
    bdml_hier_vi_rd(Y, D, X)

ReverseDiff-optimized version of bdml_hier_vi for large p.

Uses tight @inbounds loops for compact ReverseDiff tape.
Best for: p > 50 with AutoReverseDiff(; compile=true)
"""
@model function bdml_hier_vi_rd(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Hierarchical priors - explicitly typed
    σ2_δ ~ InverseGamma(T(2), T(2))
    σ2_γ ~ InverseGamma(T(2), T(2))

    θ_Y ~ MvNormal(zeros(T, p), σ2_δ * I)
    θ_D ~ MvNormal(zeros(T, p), σ2_γ * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1

    # Construct covariance matrix elements
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute means
    μ_Y = X * θ_Y
    μ_D = X * θ_D

    # TIGHT LOOP for ReverseDiff efficiency
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    # Guard against numerical issues with log
    log_det = log(max(det_Σ, eps(T)))
    const_term = -T(0.5) * (2 * log(T(2π)) + log_det)

    ll = zero(T)
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        ll += const_term - T(0.5) * mahal
    end

    Turing.@addlogprob! ll
end
