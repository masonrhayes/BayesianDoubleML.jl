# VI-Compatible BDML Models
# These models use a direct correlation parameterization (ρ ~ Beta)
# instead of LKJCholesky to ensure compatibility with Turing's ADVI

"""
    bdml_basic_vi(Y, D, X)

Basic BDML model with VI-compatible correlation parameterization.

This version uses a direct ρ ~ Beta(2, 2) prior instead of LKJCholesky,
which is compatible with Turing's ADVI variational inference.

# Model Specification (DiTraglia & Liu 2025, Section 4)

**Reduced form equations:**
    Y = X'δ + U,    [U; V]|X ~ N(0, Σ)  (Eq. 13)
    D = X'γ + V,                        (Eq. 5)

**Priors:**
- δ ~ N(0, 25*I_p)      [Outcome coefficients, Eq. 12]
- γ ~ N(0, 25*I_p)      [Treatment coefficients]
- σ_U ~ Cauchy⁺(0, 2.5) [Outcome error scale]
- σ_V ~ Cauchy⁺(0, 2.5) [Treatment error scale]
- ρ ~ Beta(2, 2)        [Correlation mapped to [-1, 1]]

**Correlation parameterization:**
The correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1]. This VI-compatible 
parameterization avoids the LKJCholesky structure.

**Causal effect recovery:**
    α = ρ·σ_U / σ_V   (Eq. 15)

# Notes
- This model is identical to bdml_basic() statistically, but uses a 
  correlation parameterization compatible with ADVI
- For causal inference, α = ρ·σ_U / σ_V is computed post-hoc from posteriors
- This implements BDML-Basic variation of Algorithm 1

See also: `bdml_basic`, `bdml_hier_vi`
"""
@model function bdml_basic_vi(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Priors on reduced form coefficients (Eq. 12, 5)
    δ ~ MvNormal(zeros(T, p), T(25.0) * I)
    γ ~ MvNormal(zeros(T, p), T(25.0) * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation: Beta(2, 2) on [0, 1], then map to [-1, 1]
    # Using Beta(2, 2) gives a distribution centered at 0.5 with moderate spread
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1  # Map from [0, 1] to [-1, 1]

    # Construct covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

    # Compute log-likelihood using bivariate normal density (Equation 13)
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = zero(T)
    @inbounds for i in 1:n
        # Reduced form residuals (Eq. 12, 5)
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

# Model Specification (DiTraglia & Liu 2025, Section 4)

**Reduced form equations:**
    Y = X'δ + U,    [U; V]|X ~ N(0, Σ)  (Eq. 13)
    D = X'γ + V,                        (Eq. 5)

**Hierarchical priors:**
- σ²_δ ~ InvGamma(2, 2)  [Variance hyperparameter for δ]
- σ²_γ ~ InvGamma(2, 2)  [Variance hyperparameter for γ]
- δ ~ N(0, σ²_δ*I_p)    [Outcome coefficients with adaptive shrinkage]
- γ ~ N(0, σ²_γ*I_p)    [Treatment coefficients with adaptive shrinkage]
- σ_U ~ Cauchy⁺(0, 2.5)  [Outcome error scale]
- σ_V ~ Cauchy⁺(0, 2.5)  [Treatment error scale]
- ρ ~ Beta(2, 2)        [Correlation mapped to [-1, 1]]

**Hierarchical structure:**
The hierarchical prior is equivalent to placing independent Student-t(4) 
distributions on each coefficient marginally. This provides adaptive 
shrinkage that learns the appropriate regularization from the data.

**Correlation parameterization:**
ρ_raw ~ Beta(2, 2) on [0, 1], then transformed to ρ = 2*ρ_raw - 1.

**Causal effect recovery:**
    α = ρ·σ_U / σ_V   (Eq. 15)

# Notes
- This model is identical to bdml_hier() statistically, but uses a 
  correlation parameterization compatible with ADVI
- For causal inference, α = ρ·σ_U / σ_V is computed post-hoc from posteriors
- This implements BDML-Hier variation of Algorithm 1
- Provides adaptive shrinkage compared to BDML-Basic

See also: `bdml_hier`, `bdml_basic_vi`
"""
@model function bdml_hier_vi(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Hierarchical hyperpriors (Section 6, Table 1)
    σ²_δ ~ InverseGamma(T(2), T(2))
    σ²_γ ~ InverseGamma(T(2), T(2))

    # Priors on reduced form coefficients with hierarchical structure (Eq. 12, 5)
    δ ~ MvNormal(zeros(T, p), σ²_δ * I)
    γ ~ MvNormal(zeros(T, p), σ²_γ * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation: Beta(2, 2) on [0, 1], then map to [-1, 1]
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1  # Map from [0, 1] to [-1, 1]

    # Construct covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

    # Compute log-likelihood using bivariate normal density (Equation 13)
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = zero(T)
    @inbounds for i in 1:n
        # Reduced form residuals (Eq. 12, 5)
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

Implements BDML-Basic variation of Algorithm 1 with paper notation.

See: `bdml_basic_vi`, `bdml_basic`
"""
@model function bdml_basic_vi_rd(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Priors - explicitly typed for type stability
    δ ~ MvNormal(zeros(T, p), T(25.0) * I)
    γ ~ MvNormal(zeros(T, p), T(25.0) * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1

    # Construct covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

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
        # Reduced form residuals (Eq. 12, 5)
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
Implements BDML-Hier variation of Algorithm 1 with paper notation.

See: `bdml_hier_vi`, `bdml_hier`
"""
@model function bdml_hier_vi_rd(Y, D, X)
    n, p = size(X)
    T = eltype(X)

    # Hierarchical hyperpriors - explicitly typed
    σ²_δ ~ InverseGamma(T(2), T(2))
    σ²_γ ~ InverseGamma(T(2), T(2))

    # Priors on reduced form coefficients with hierarchical structure
    δ ~ MvNormal(zeros(T, p), σ²_δ * I)
    γ ~ MvNormal(zeros(T, p), σ²_γ * I)

    σ_U ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))
    σ_V ~ truncated(Cauchy(T(0), T(2.5)), T(0.1), T(Inf))

    # VI-compatible correlation
    ρ_raw ~ Beta(T(2), T(2))
    ρ = 2 * ρ_raw - 1

    # Construct covariance matrix Σ (Equation 14)
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

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
        # Reduced form residuals (Eq. 12, 5)
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        ll += const_term - T(0.5) * mahal
    end

    Turing.@addlogprob! ll
end
