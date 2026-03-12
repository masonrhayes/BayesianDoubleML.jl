"""
    bdml_basic(Y, D, X)

Bayesian Double Machine Learning - Basic Model.

Implements the BDML-Basic variation of Algorithm 1 from DiTraglia & Liu (2025), 
Section 4. This model uses fixed prior variances on the reduced form coefficients.

# The Bivariate Reduced Form Model

The BDML approach avoids regularization-induced confounding by using a 
bivariate reduced form parameterization (Section 4, Equations 13-15):

**Outcome equation:**
    ``Y = X'\\delta + U``          (Eq. 12)

**Treatment equation:**
    ``D = X'\\gamma + V``          (Eq. 5)

**Joint error distribution:**
    ``[U; V] | X \\sim N(0, \\Sigma)``  (Eq. 13)

where ``\\Sigma`` is the ``2\\times2`` covariance matrix:
    ``\\Sigma = [\\sigma^2_U, \\sigma_{UV}; \\sigma_{UV}, \\sigma^2_V]``

**Causal effect recovery:**
The causal effect ``\\alpha`` is recovered from the error covariance via:
    ``\\alpha = \\text{Cov}(U, V) / \\text{Var}(V) = \\sigma_{UV} / \\sigma^2_V = \\rho \\cdot \\sigma_U / \\sigma_V``   (Eq. 15)

This works because ``U = \\epsilon + \\alpha V`` (Eq. 6), where ``\\epsilon`` is the structural error 
uncorrelated with V. Therefore: ``\\text{Cov}(U, V) = \\text{Cov}(\\epsilon + \\alpha V, V) = \\alpha \\cdot \\text{Var}(V)``.

# This Variation: BDML-Basic

**Priors on reduced form coefficients:**
- ``\\delta \\sim N(0, 25 \\cdot I_p)``    [``\\text{Normal}(0, 5^2)`` for outcome coefficients]
- ``\\gamma \\sim N(0, 25 \\cdot I_p)``    [``\\text{Normal}(0, 5^2)`` for treatment coefficients]

These are fixed-variance priors that provide uniform shrinkage across all 
covariates. This corresponds to "BDML-Basic" in paper Section 6, Table 1.

**Priors on error covariance:**
- ``\\sigma_U \\sim \\text{Cauchy}^+(0, 2.5)``   [Half-Cauchy for outcome error scale]
- ``\\sigma_V \\sim \\text{Cauchy}^+(0, 2.5)``   [Half-Cauchy for treatment error scale]  
- ``R \\sim \\text{LKJ}(4)``              [LKJ prior with ``\\eta=4`` for correlation matrix]

The correlation parameter ``\\rho`` is extracted from the LKJCholesky factor ``R_\\text{chol}.L[2,1]``.

For adaptive shrinkage that learns the appropriate regularization from data,
use `bdml_hier()` (BDML-Hierarchical) instead.

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.

See also: `bdml_hier`, `bdml_basic_vi`
"""
@model function bdml_basic(Y, D, X)
    n, p = size(X)

    # Priors on reduced form coefficients (Equation 12, 5)
    # δ: coefficients for Y = X'δ + U
    # γ: coefficients for D = X'γ + V
    δ ~ MvNormal(zeros(p), 25.0 * I)
    γ ~ MvNormal(zeros(p), 25.0 * I)

    # Priors on error scales (Section 6, p. 25)
    σ_V ~ truncated(Cauchy(0, 2.5), 0.01, Inf)
    σ_U ~ truncated(Cauchy(0, 2.5), 0.01, Inf)

    # Prior on correlation matrix - LKJ(4) as specified in paper
    R_chol ~ LKJCholesky(2, 4)

    # Extract correlation coefficient ρ from Cholesky factor (Equation 14)
    # For a 2×2 correlation matrix R = L * L', where L is lower triangular:
    # L = [1        0
    #      ρ  sqrt(1-ρ²)]
    # Therefore: ρ = L[2,1]
    L = R_chol.L
    ρ = L[2, 1]

    # Construct covariance matrix Σ (Equation 14)
    # Σ_11 = Var(U) = σ²_U
    # Σ_22 = Var(V) = σ²_V
    # Σ_12 = Cov(U,V) = ρ·σ_U·σ_V
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

    # Compute log-likelihood using bivariate normal density (Equation 13)
    # For bivariate normal: log p(Y,D|X) = Σᵢ -0.5*(2log(2π) + log|Σ| + rᵢ'Σ⁻¹rᵢ)
    # where rᵢ = [Yᵢ - Xᵢ'δ; Dᵢ - Xᵢ'γ] = [Uᵢ; Vᵢ]
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = 0.0
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood contribution (using NaNMath for numerical stability)
        ll += -0.5 * (2 * log(2π) + NaNMath.log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end

"""
    bdml_hier(Y, D, X)

Bayesian Double Machine Learning - Hierarchical Model.

Implements the BDML-Hier variation of Algorithm 1 from DiTraglia & Liu (2025), 
Section 4. This model uses adaptive hierarchical priors for data-driven 
shrinkage on the reduced form coefficients.

# The Bivariate Reduced Form Model

The BDML approach avoids regularization-induced confounding by using a 
bivariate reduced form parameterization (Section 4, Equations 13-15):

**Outcome equation:**
    ``Y = X'\\delta + U``          (Eq. 12)

**Treatment equation:**
    ``D = X'\\gamma + V``          (Eq. 5)

**Joint error distribution:**
    ``[U; V] | X \\sim N(0, \\Sigma)``  (Eq. 13)

where ``\\Sigma`` is the ``2\\times2`` covariance matrix:
    ``\\Sigma = [\\sigma^2_U, \\sigma_{UV}; \\sigma_{UV}, \\sigma^2_V]``

**Causal effect recovery:**
The causal effect ``\\alpha`` is recovered from the error covariance via:
    ``\\alpha = \\text{Cov}(U, V) / \\text{Var}(V) = \\sigma_{UV} / \\sigma^2_V = \\rho \\cdot \\sigma_U / \\sigma_V``   (Eq. 15)

This works because ``U = \\epsilon + \\alpha V`` (Eq. 6), where ``\\epsilon`` is the structural error 
uncorrelated with V. Therefore: ``\\text{Cov}(U, V) = \\text{Cov}(\\epsilon + \\alpha V, V) = \\alpha \\cdot \\text{Var}(V)``.

# This Variation: BDML-Hier

**Hierarchical priors on reduced form coefficients:**
The key difference from BDML-Basic is the adaptive prior structure:

1. Hyperpriors on variance hyperparameters:
   - ``\\sigma^2_\\delta \\sim \\text{InvGamma}(2, 2)``
   - ``\\sigma^2_\\gamma \\sim \\text{InvGamma}(2, 2)``

2. Conditional priors on coefficients:
   - ``\\delta | \\sigma^2_\\delta \\sim N(0, \\sigma^2_\\delta \\cdot I_p)``
   - ``\\gamma | \\sigma^2_\\gamma \\sim N(0, \\sigma^2_\\gamma \\cdot I_p)``

**Why this works:**
This hierarchical structure is equivalent to placing independent ``\\text{Student-}t(4)`` 
distributions on each coefficient ``\\delta_j`` and ``\\gamma_j``. The ``\\text{InvGamma}(2, 2)`` hyperprior 
induces heavy tails that allow some coefficients to escape shrinkage while 
regularizing others. This provides adaptive shrinkage that learns the 
appropriate regularization level from the data (Section 6, Table 1).

**Priors on error covariance:**
- ``\\sigma_U \\sim \\text{Cauchy}^+(0, 2.5)``   [Half-Cauchy for outcome error scale]
- ``\\sigma_V \\sim \\text{Cauchy}^+(0, 2.5)``   [Half-Cauchy for treatment error scale]  
- ``R \\sim \\text{LKJ}(4)``              [LKJ prior with ``\\eta=4`` for correlation matrix]

The correlation parameter ``\\rho`` is extracted from the LKJCholesky factor ``R_\\text{chol}.L[2,1]``.

**Recommendation:** The paper's simulation study (Section 6, Table 1) shows 
BDML-Hier achieves better coverage (0.94) compared to BDML-Basic (0.91-0.93), 
making it the recommended default choice for most applications.

For fixed shrinkage with simpler interpretation, use `bdml_basic()` instead.

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.

See also: `bdml_basic`, `bdml_hier_vi`
"""
@model function bdml_hier(Y, D, X)
    n, p = size(X)

    # Hierarchical hyperpriors on variance hyperparameters (Section 6, p. 25)
    # σ²_δ governs shrinkage of outcome coefficients δ
    # σ²_γ governs shrinkage of treatment coefficients γ
    σ²_δ ~ InverseGamma(2, 2)
    σ²_γ ~ InverseGamma(2, 2)

    # Priors on reduced form coefficients with hierarchical structure (Eq. 12, 5)
    # δ: coefficients for Y = X'δ + U
    # γ: coefficients for D = X'γ + V
    # These are equivalent to Student-t(4) priors marginally
    δ ~ MvNormal(zeros(p), σ²_δ * I)
    γ ~ MvNormal(zeros(p), σ²_γ * I)

    # Priors on error scales (Section 6, p. 25)
    σ_U ~ truncated(Cauchy(0, 2.5), 0.01, Inf)
    σ_V ~ truncated(Cauchy(0, 2.5), 0.01, Inf)

    # Prior on correlation matrix - LKJ(4) as specified in paper
    R_chol ~ LKJCholesky(2, 4)

    # Extract correlation coefficient ρ from Cholesky factor (Equation 14)
    # For a 2×2 correlation matrix R = L * L', where L is lower triangular:
    # L = [1        0
    #      ρ  sqrt(1-ρ²)]
    # Therefore: ρ = L[2,1]
    L = R_chol.L
    ρ = L[2, 1]

    # Construct covariance matrix Σ (Equation 14)
    # Σ_11 = Var(U) = σ²_U
    # Σ_22 = Var(V) = σ²_V
    # Σ_12 = Cov(U,V) = ρ·σ_U·σ_V
    Σ_11 = σ_U^2
    Σ_22 = σ_V^2
    Σ_12 = ρ * σ_U * σ_V

    # Compute reduced form means
    μ_Y = X * δ
    μ_D = X * γ

    # Compute log-likelihood using bivariate normal density (Equation 13)
    # For bivariate normal: log p(Y,D|X) = Σᵢ -0.5*(2log(2π) + log|Σ| + rᵢ'Σ⁻¹rᵢ)
    # where rᵢ = [Yᵢ - Xᵢ'δ; Dᵢ - Xᵢ'γ] = [Uᵢ; Vᵢ]
    det_Σ = Σ_11 * Σ_22 - Σ_12^2
    inv_Σ_11 = Σ_22 / det_Σ
    inv_Σ_22 = Σ_11 / det_Σ
    inv_Σ_12 = -Σ_12 / det_Σ

    ll = 0.0
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood contribution (using NaNMath for numerical stability)
        ll += -0.5 * (2 * log(2π) + NaNMath.log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end

export bdml_basic, bdml_hier
