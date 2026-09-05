# Collapsed BDML sufficient statistics and exact conditional for (delta, gamma)

using LinearAlgebra
using Random

"""
    CollapsedBDML

Rank-aware sufficient statistics for the BDML likelihood after analytically
integrating out `(delta, gamma)`.
"""
struct CollapsedBDML
    lambda::Vector{Float64}
    Q::Matrix{Float64}
    G::Matrix{Float64}
    projected_W::Matrix{Float64}
    residual_WtW::Matrix{Float64}
    rank::Int
    n::Int
    p::Int
    model_type::Symbol
    Ysd::Float64
    Dsd::Float64
end

"""
    CollapsedBDML(model::AbstractBDMLModel)

Build rank-aware collapsed statistics from a standardized `BDMLModel`.
"""
function CollapsedBDML(model::AbstractBDMLModel)
    n = nobs(model)
    p = ncovariates(model)
    W = hcat(model.Y, model.D)

    F = svd(model.X; full = true)
    singular_tolerance = isempty(F.S) ? 0.0 : max(n, p) * eps(Float64) * maximum(F.S)
    r = count(>(singular_tolerance), F.S)

    Q = Matrix(F.V)
    lambda = zeros(p)
    G = zeros(p, 2)
    projected_W = zeros(r, 2)
    if r > 0
        U_r = @view F.U[:, 1:r]
        projected_W .= U_r' * W
        lambda[1:r] .= (@view F.S[1:r]) .^ 2
        G[1:r, :] .= (@view F.S[1:r]) .* projected_W
        residual_W = W - U_r * projected_W
    else
        residual_W = W
    end

    residual_WtW = Matrix(Symmetric(residual_W' * residual_W))
    return CollapsedBDML(
        lambda, Q, G, projected_W, residual_WtW, r, n, p,
        model_type(model), model.stats.Y_sd, model.stats.D_sd,
    )
end

@inline _prior_variance(::CollapsedBDML, ::Real, ::Real, ::Val{:basic}) = (25.0, 25.0)
@inline _prior_variance(::CollapsedBDML, s2d::Real, s2g::Real, ::Val{:hier}) = (s2d, s2g)
@inline _prior_variance(c::CollapsedBDML, s2d::Real, s2g::Real) =
    _prior_variance(c, s2d, s2g, Val(c.model_type))

"""
    log_marginal_collapsed(c, sigma_U, sigma_V, rho, s2d, s2g)

Compute `log p(W | Sigma, s2d, s2g)` in `O(rank(X))`. The calculation uses
the SVD observation-space marginal directly, avoiding cancellation between
large precision-space terms when `X` is rank deficient or `p` is close to `n`.
"""
function log_marginal_collapsed(
        c::CollapsedBDML,
        sigma_U::Real,
        sigma_V::Real,
        rho::Real,
        s2d::Real,
        s2g::Real,
    )
    sigma_U2 = sigma_U^2
    sigma_V2 = sigma_V^2
    sigma_UV = rho * sigma_U * sigma_V
    one_minus_rho2 = 1 - rho^2
    det_Sigma = sigma_U2 * sigma_V2 * one_minus_rho2
    det_Sigma > 0 || return -Inf

    prior_delta, prior_gamma = _prior_variance(c, s2d, s2g)
    (prior_delta > 0 && prior_gamma > 0) || return -Inf

    residual_quadratic = (
        sigma_V2 * c.residual_WtW[1, 1] -
            2 * sigma_UV * c.residual_WtW[1, 2] +
            sigma_U2 * c.residual_WtW[2, 2]
    ) / det_Sigma
    log_determinants = (c.n - c.rank) * log(det_Sigma)
    quadratic = residual_quadratic

    @inbounds for j in 1:c.rank
        lambda_j = c.lambda[j]
        a = sigma_U2 + lambda_j * prior_delta
        b = sigma_UV
        d = sigma_V2 + lambda_j * prior_gamma
        det_j = det_Sigma +
            lambda_j * (prior_delta * sigma_V2 + prior_gamma * sigma_U2) +
            lambda_j^2 * prior_delta * prior_gamma
        z1 = c.projected_W[j, 1]
        z2 = c.projected_W[j, 2]
        log_determinants += log(det_j)
        quadratic += (d * z1^2 - 2 * b * z1 * z2 + a * z2^2) / det_j
    end

    value = -c.n * log(2pi) - 0.5 * (log_determinants + quadratic)
    return isfinite(value) ? value : -Inf
end

function log_marginal_collapsed(
        c::CollapsedBDML, Sigma::AbstractMatrix, s2d::Real, s2g::Real,
    )
    sigma_U2 = Sigma[1, 1]
    sigma_V2 = Sigma[2, 2]
    (sigma_U2 > 0 && sigma_V2 > 0) || return -Inf
    sigma_U = sqrt(sigma_U2)
    sigma_V = sqrt(sigma_V2)
    rho = Sigma[1, 2] / (sigma_U * sigma_V)
    return log_marginal_collapsed(c, sigma_U, sigma_V, rho, s2d, s2g)
end

log_marginal_collapsed(c::CollapsedBDML, Sigma::AbstractMatrix) =
    log_marginal_collapsed(c, Sigma, 25.0, 25.0)

"""
    draw_coefficients_collapsed(c, sigma_U, sigma_V, rho, s2d, s2g, rng)

Draw `(delta, gamma)` from their exact joint Gaussian conditional posterior.
"""
function draw_coefficients_collapsed(
        c::CollapsedBDML,
        sigma_U::Real,
        sigma_V::Real,
        rho::Real,
        s2d::Real,
        s2g::Real,
        rng::AbstractRNG,
    )
    sigma_U2 = sigma_U^2
    sigma_V2 = sigma_V^2
    det_Sigma = sigma_U2 * sigma_V2 * (1 - rho^2)
    omega11 = sigma_V2 / det_Sigma
    omega22 = sigma_U2 / det_Sigma
    omega12 = -rho * sigma_U * sigma_V / det_Sigma
    prior_delta, prior_gamma = _prior_variance(c, s2d, s2g)
    pd = inv(prior_delta)
    pg = inv(prior_gamma)

    delta_eigen = Vector{Float64}(undef, c.p)
    gamma_eigen = similar(delta_eigen)
    @inbounds for j in 1:c.p
        lambda_j = c.lambda[j]
        a = lambda_j * omega11 + pd
        b = lambda_j * omega12
        d = lambda_j * omega22 + pg
        det_j = lambda_j^2 / det_Sigma +
            lambda_j * (omega11 * pg + omega22 * pd) + pd * pg
        covariance11 = d / det_j
        covariance22 = a / det_j
        covariance12 = -b / det_j

        g1 = c.G[j, 1]
        g2 = c.G[j, 2]
        h1 = g1 * omega11 + g2 * omega12
        h2 = g1 * omega12 + g2 * omega22
        mean1 = covariance11 * h1 + covariance12 * h2
        mean2 = covariance12 * h1 + covariance22 * h2

        L = cholesky(
            Symmetric(
                [
                    covariance11 covariance12
                    covariance12 covariance22
                ]
            )
        ).L
        z1, z2 = randn(rng), randn(rng)
        delta_eigen[j] = mean1 + L[1, 1] * z1
        gamma_eigen[j] = mean2 + L[2, 1] * z1 + L[2, 2] * z2
    end
    return c.Q * delta_eigen, c.Q * gamma_eigen
end
