@model function bdml_basic(Y, D, X)
    n, p = size(X)

    θ_Y ~ MvNormal(zeros(p), 25.0 * I)
    θ_D ~ MvNormal(zeros(p), 25.0 * I)

    σ_U ~ truncated(Cauchy(0, 2.5), 0.1, Inf)
    σ_V ~ truncated(Cauchy(0, 2.5), 0.1, Inf)

    # Use LKJCholesky for correlation matrix as specified in paper
    R_chol ~ LKJCholesky(2, 4)

    # Extract correlation coefficient from Cholesky factor
    # For a 2x2 correlation matrix R = L * L', where L is lower triangular:
    # L = [1        0
    #      ρ  sqrt(1-ρ²)]
    # So ρ = L[2,1]
    L = R_chol.L
    ρ = L[2, 1]

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

    ll = 0.0
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood for bivariate normal (using NaNMath for numerical stability)
        ll += -0.5 * (2 * log(2π) + NaNMath.log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end

@model function bdml_hier(Y, D, X)
    n, p = size(X)

    σ2_δ ~ InverseGamma(2, 2)
    σ2_γ ~ InverseGamma(2, 2)

    θ_Y ~ MvNormal(zeros(p), σ2_δ * I)
    θ_D ~ MvNormal(zeros(p), σ2_γ * I)

    σ_U ~ truncated(Cauchy(0, 2.5), 0.1, Inf)
    σ_V ~ truncated(Cauchy(0, 2.5), 0.1, Inf)

    # Use LKJCholesky for correlation matrix as specified in paper
    R_chol ~ LKJCholesky(2, 4)

    # Extract correlation coefficient from Cholesky factor
    L = R_chol.L
    ρ = L[2, 1]

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

    ll = 0.0
    @inbounds for i in 1:n
        U_i = Y[i] - μ_Y[i]
        V_i = D[i] - μ_D[i]

        # Mahalanobis distance: [U, V]' * Σ^(-1) * [U, V]
        mahal = inv_Σ_11 * U_i^2 + 2 * inv_Σ_12 * U_i * V_i + inv_Σ_22 * V_i^2

        # Log-likelihood for bivariate normal (using NaNMath for numerical stability)
        ll += -0.5 * (2 * log(2π) + NaNMath.log(det_Σ) + mahal)
    end

    Turing.@addlogprob! ll
end
