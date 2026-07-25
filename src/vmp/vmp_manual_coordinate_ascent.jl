# Conjugate VMP via manual coordinate ascent (sufficient-statistics form).

"""
    VMPManualState{T,V<:AbstractVector{T},M<:AbstractMatrix{T},QΣ,Qτ}

Parametric state struct for the conjugate VMP coordinate-ascent algorithm.

# Type parameters
- `T`: scalar numeric type (typically `Float64`)
- `V`: concrete vector type for coefficient means
- `M`: concrete matrix type for coefficient covariances
- `QΣ`: concrete type of the variational posterior for the ``2 \times 2`` error covariance ``\\Sigma``
- `Qτ`: concrete type of the variational posteriors for precision hyperparameters ``\\tau_\\delta`` and ``\\tau_\\gamma`.
  `Nothing` for the basic model (no hyperparameters); `Gamma{Float64}` for the hierarchical model.

# Fields
- `mδ::V`: variational mean of outcome coefficients ``\\delta``
- `mγ::V`: variational mean of treatment coefficients ``\\gamma``
- `Vδ::M`: variational covariance of ``\\delta``
- `Vγ::M`: variational covariance of ``\\gamma``
- `qΣ::QΣ`: variational posterior ``q(\\Sigma)`` as `InverseWishart`
- `τδ::Qτ`: variational posterior ``q(\\tau_\\delta)`` (or `nothing` for the basic model)
- `τγ::Qτ`: variational posterior ``q(\\tau_\\gamma)`` (or `nothing` for the basic model)

# Notes
For a `BDMLBasicModel` `Qτ = Nothing`; for a `BDMLHierarchicalModel` `Qτ = Gamma{Float64}`.
"""
struct VMPManualState{T,V<:AbstractVector{T},M<:AbstractMatrix{T},QΣ,Qτ}
    mδ::V
    mγ::V
    Vδ::M
    Vγ::M
    qΣ::QΣ
    τδ::Qτ
    τγ::Qτ
end

"""
    _logmultigamma(a::Real, d::Int)

Multivariate log-gamma function ``\\log\\gamma_d(a)`` for a `d`-dimensional Wishart / InverseWishart.

# Formula
``\\log\\gamma_d(a) = \\frac{d(d-1)}{4}\\log\\pi + \\sum_{j=1}^{d} \\log\\gamma\\!\\left(a + \frac{1-j}{2}\right)``
"""
function _logmultigamma(a::Real, d::Int)
    return d * (d - 1) / 4 * log(pi) + sum(
        Distributions.loggamma(a + (1 - j) / 2) for j in 1:d
    )
end

"""
    _expected_logdet(qΣ::InverseWishart)

Expected log-determinant ``\\mathbb{E}_{q(\\Sigma)}[\\log|\\Sigma|]`` under an `InverseWishart` distribution.

# Formula
For ``q(\\Sigma) = \text{InverseWishart}(\nu, S)`` with dimension ``d``:
``\\mathbb{E}[\\log|\\Sigma|] = \\log|S| - d\\log 2 - \\sum_{j=1}^{d} \\psi\\!\\left(\frac{\nu+1-j}{2}\right)``
where ``\\psi`` is the digamma function.
"""
function _expected_logdet(qΣ::InverseWishart)
    ν, S = Distributions.params(qΣ)
    d = size(S, 1)
    return logdet(Symmetric(S)) - d * log(2) - sum(
        Distributions.digamma((ν + 1 - j) / 2) for j in 1:d
    )
end

"""
    _inversewishart_entropy(qΣ::InverseWishart)

Differential entropy of an `InverseWishart` distribution.

# Formula
For ``q(\\Sigma) = \text{InverseWishart}(\nu, S)`` with dimension ``d``:
``
H(q) = -\frac{\nu}{2}\\log|S| + \frac{\nu d}{2}\\log 2 + \\log\\gamma_d\\!\\left(\frac{\nu}{2}\right)
       + \frac{\nu+d+1}{2}\\mathbb{E}[\\log|\\Sigma|] + \frac{\nu d}{2}
``
where ``\\mathbb{E}[\\log|\\Sigma|]`` is computed by [`_expected_logdet`](@ref).
"""
function _inversewishart_entropy(qΣ::InverseWishart)
    ν, S = Distributions.params(qΣ)
    d = size(S, 1)
    elogdet = _expected_logdet(qΣ)
    return -ν / 2 * logdet(Symmetric(S)) + ν * d / 2 * log(2) +
        _logmultigamma(ν / 2, d) + (ν + d + 1) / 2 * elogdet + ν * d / 2
end

"""
    _vmp_elbo_core(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, R::Matrix{Float64},
        logdet_Vδ::Float64, logdet_Vγ::Float64,
    )

Shared ELBO terms that are identical for both the basic and hierarchical models.

Computes three components:

1. **Data log-likelihood**
   ``-\frac{n d}{2}\\log(2\\pi) - \frac{n}{2}\\mathbb{E}[\\log|\\Sigma|] - \frac{1}{2}\text{tr}(\\Omega R)``
2. **Inverse-Wishart prior** on ``\\Sigma``
3. **Entropy contributions** from ``q(\\delta)``, ``q(\\gamma)``, and ``q(\\Sigma)``

# Arguments
- `state`: current [`VMPManualState`](@ref)
- `Sxx`, `xsy`, `xsd`, `Sww`: sufficient statistics
- `n`: number of observations
- `ν0`, `S0_mat`: prior hyperparameters for ``q(\\Sigma)``
- `R`: residual cross-product matrix (pre-computed by [`_vmp_step_shared`](@ref))
- `logdet_Vδ`, `logdet_Vγ`: log-determinants of the current coefficient covariances,
  cached from the Cholesky factor to avoid redundant ``O(p^3)`` work

# Returns
The scalar ELBO contribution shared across model variants.
"""
function _vmp_elbo_core(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, R::Matrix{Float64},
        logdet_Vδ::Float64, logdet_Vγ::Float64,
    )
    ν, S = Distributions.params(state.qΣ)
    d = size(S, 1)
    Ω = ν * inv(Symmetric(S))
    Ω = Matrix(Symmetric(Ω))
    elogdetΣ = _expected_logdet(state.qΣ)

    elbo = -n * d / 2 * log(2 * pi) - n / 2 * elogdetΣ - tr(Ω * R) / 2

    ν0_prior = ν0 / 2 * logdet(Symmetric(S0_mat)) - ν0 * d / 2 * log(2) -
        _logmultigamma(ν0 / 2, d) - (ν0 + d + 1) / 2 * elogdetΣ - tr(S0_mat * Ω) / 2
    elbo += ν0_prior

    elbo += (length(state.mδ) * (1 + log(2 * pi)) + logdet_Vδ) / 2
    elbo += (length(state.mγ) * (1 + log(2 * pi)) + logdet_Vγ) / 2
    elbo += _inversewishart_entropy(state.qΣ)
    return elbo
end

"""
    _vmp_elbo(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLBasicModel, ::VMPMethod, cache,
    )

ELBO for the basic BDML model with fixed Gaussian priors ``\\delta, \\gamma \\sim N(0, 25 I_p)``.

Calls [`_vmp_elbo_core`](@ref) and adds the fixed-prior terms
``-\frac{p}{2}\\log(2\\pi \\cdot 25) - \frac{1}{50}(\\|m_\\delta\\|^2 + \text{tr}(V_\\delta))``
(and similarly for ``\\gamma``).
"""
function _vmp_elbo(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLBasicModel, ::VMPMethod, cache,
    )
    R, logdet_Vδ, logdet_Vγ = cache.R, cache.logdet_Vδ, cache.logdet_Vγ
    elbo = _vmp_elbo_core(state, Sxx, xsy, xsd, Sww, n, ν0, S0_mat, R, logdet_Vδ, logdet_Vγ)
    Qδ = dot(state.mδ, state.mδ) + tr(state.Vδ)
    Qγ = dot(state.mγ, state.mγ) + tr(state.Vγ)
    elbo += -length(state.mδ) / 2 * log(2 * pi * 25.0) - Qδ / 50
    elbo += -length(state.mγ) / 2 * log(2 * pi * 25.0) - Qγ / 50
    return elbo
end

"""
    _vmp_elbo(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLHierarchicalModel, method::VMPMethod, cache,
    )

ELBO for the hierarchical BDML model with adaptive shrinkage.

Calls [`_vmp_elbo_core`](@ref) and adds the Gamma hyperprior terms for
``\\tau_\\delta, \tau_\\gamma \\sim \text{Gamma}(a_\\tau, b_\\tau)`` and their
variational posteriors ``q(\\tau_\\delta)``, ``q(\\tau_\\gamma)``.
"""
function _vmp_elbo(
        state::VMPManualState, Sxx, xsy, xsd, Sww, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLHierarchicalModel, method::VMPMethod, cache,
    )
    R, logdet_Vδ, logdet_Vγ = cache.R, cache.logdet_Vδ, cache.logdet_Vγ
    elbo = _vmp_elbo_core(state, Sxx, xsy, xsd, Sww, n, ν0, S0_mat, R, logdet_Vδ, logdet_Vγ)
    Qδ = dot(state.mδ, state.mδ) + tr(state.Vδ)
    Qγ = dot(state.mγ, state.mγ) + tr(state.Vγ)
    for (m, V, qτ, Q) in ((state.mδ, state.Vδ, state.τδ, Qδ), (state.mγ, state.Vγ, state.τγ, Qγ))
        A, θ = Distributions.params(qτ)
        Eτ = A * θ
        Elogτ = Distributions.digamma(A) + log(θ)
        elbo += -length(m) / 2 * log(2 * pi) + length(m) / 2 * Elogτ - Eτ * Q / 2
        elbo += (method.aτ - 1) * Elogτ - Eτ / method.bτ -
            Distributions.loggamma(method.aτ) - method.aτ * log(method.bτ)
        elbo += A + log(θ) + Distributions.loggamma(A) + (1 - A) * Distributions.digamma(A)
    end
    return elbo
end

"""
    _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLBasicModel, ::VMPMethod)

Create the initial [`VMPManualState`](@ref) for a basic model.

Initialises ``m_\\delta = m_\\gamma = 0``, ``V_\\delta = V_\\gamma = 25 I_p``,
and ``q(\\Sigma) = \text{InverseWishart}(\nu_0, S_0)``.
"""
function _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLBasicModel, ::VMPMethod)
    mδ = zeros(p)
    mγ = zeros(p)
    Vδ = Matrix{Float64}(I, p, p) * 25.0
    Vγ = copy(Vδ)
    qΣ = InverseWishart(ν0, S0_mat)
    return VMPManualState(mδ, mγ, Vδ, Vγ, qΣ, nothing, nothing)
end

"""
    _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLHierarchicalModel, method::VMPMethod)

Create the initial [`VMPManualState`](@ref) for a hierarchical model.

Same as the basic initialisation but also sets
``q(\\tau_\\delta) = q(\\tau_\\gamma) = \text{Gamma}(a_\\tau, b_\\tau)``.
"""
function _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLHierarchicalModel, method::VMPMethod)
    mδ = zeros(p)
    mγ = zeros(p)
    Vδ = Matrix{Float64}(I, p, p) * 25.0
    Vγ = copy(Vδ)
    qΣ = InverseWishart(ν0, S0_mat)
    τδ = Gamma(method.aτ, method.bτ)
    τγ = Gamma(method.aτ, method.bτ)
    return VMPManualState(mδ, mγ, Vδ, Vγ, qΣ, τδ, τγ)
end

"""
    _vmp_step_shared(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν::Float64, S0_mat::Matrix{Float64},
        λδ::Float64, λγ::Float64,
    )

Execute one coordinate-ascent VMP step, shared across basic and hierarchical models.

Performs three blocks:

1. **Coefficient precision matrices** (parallel mean-field):
   ``P_\\delta = \\omega_{11} S_{xx} + \\lambda_\\delta I_p``
   ``P_\\gamma = \\omega_{22} S_{xx} + \\lambda_\\gamma I_p``
   where ``\\Omega = \nu_\\Sigma \\Psi_\\Sigma^{-1}`` is the expected precision of the bivariate errors.

2. **Cholesky solve for new means**:
   ``m_\\delta^{\text{new}} = P_\\delta^{-1} h_\\delta``,
   ``m_\\gamma^{\text{new}} = P_\\gamma^{-1} h_\\gamma``
   with natural-parameter vectors that use the *old* counterpart mean
   (parallel mean-field, matching the RxInfer graph update order).

3. **Residual cross-product and ``q(\\Sigma)`` update**:
   Forms ``R`` from the new means/covariances and updates
   ``q(\\Sigma) = \text{InverseWishart}(\nu, S_0 + R)``.

# Arguments
- `state`: current [`VMPManualState`](@ref)
- `Sxx`, `xsy`, `xsd`, `Sww`: sufficient statistics
- `ν`: posterior degrees of freedom ``\\nu_0 + n``
- `S0_mat`: prior scale matrix for ``\\Sigma``
- `λδ`, `λγ`: shrinkage precisions (``1/25`` for basic, ``\\mathbb{E}[\\tau]`` for hierarchical)

# Returns
A tuple `(new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, rel_change, cache)` where
`cache` holds the residual matrix `R` and the cached log-determinants
``-\\log\\det(F_\\delta)``, ``-\\log\\det(F_\\gamma)`` for the ELBO computation.
"""
function _vmp_step_shared(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν::Float64, S0_mat::Matrix{Float64},
        λδ::Float64, λγ::Float64,
    )
    ν_Σ, Ψ_Σ = Distributions.params(state.qΣ)
    Ω = ν_Σ * inv(Symmetric(Ψ_Σ))
    Ω = Matrix(Symmetric(Ω))

    Pδ = Symmetric(Ω[1, 1] .* Sxx + λδ .* I)
    Pγ = Symmetric(Ω[2, 2] .* Sxx + λγ .* I)
    hδ = Ω[1, 1] .* xsy + Ω[1, 2] .* (xsd - Sxx * state.mγ)
    hγ = Ω[2, 2] .* xsd + Ω[1, 2] .* (xsy - Sxx * state.mδ)

    Fδ = cholesky(Pδ)
    Fγ = cholesky(Pγ)
    new_mδ = Fδ \ hδ
    new_mγ = Fγ \ hγ
    new_Vδ = inv(Fδ)
    new_Vγ = inv(Fγ)

    Aδ = new_Vδ + new_mδ * new_mδ'
    Aγ = new_Vγ + new_mγ * new_mγ'
    r11 = Sww[1, 1] - 2 * dot(xsy, new_mδ) + dot(Sxx, Aδ)
    r22 = Sww[2, 2] - 2 * dot(xsd, new_mγ) + dot(Sxx, Aγ)
    r12 = Sww[1, 2] - dot(xsy, new_mγ) - dot(xsd, new_mδ) + dot(new_mδ, Sxx * new_mγ)
    S = S0_mat + [r11 r12; r12 r22]
    S = Matrix(Symmetric(S))
    isposdef(Symmetric(S)) || throw(ArgumentError("VMP covariance update is not positive definite"))
    new_qΣ = InverseWishart(ν, S)

    change = max(
        mapreduce((a, b) -> abs(a - b), max, new_mδ, state.mδ),
        mapreduce((a, b) -> abs(a - b), max, new_mγ, state.mγ),
        mapreduce((a, b) -> abs(a - b), max, new_Vδ, state.Vδ),
        mapreduce((a, b) -> abs(a - b), max, new_Vγ, state.Vγ),
        mapreduce((a, b) -> abs(a - b), max, S, Ψ_Σ),
    )
    scale = max(1.0, maximum(abs, new_mδ), maximum(abs, new_mγ), maximum(abs, S))
    rel_change = change / scale

    R = Matrix(Symmetric([r11 r12; r12 r22]))
    cache = (R = R, logdet_Vδ = -logdet(Fδ), logdet_Vγ = -logdet(Fγ))

    return new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, rel_change, cache
end

"""
    _vmp_step(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν, S0_mat,
        ::BDMLBasicModel, ::VMPMethod,
    )

Single VMP coordinate-ascent step for the basic model.

Fixes the shrinkage precision at ``\\lambda = 1/25`` and calls [`_vmp_step_shared`](@ref).
"""
function _vmp_step(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν, S0_mat,
        ::BDMLBasicModel, ::VMPMethod,
    )
    λ = 1.0 / 25.0
    new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, rel_change, cache = _vmp_step_shared(
        state, Sxx, xsy, xsd, Sww, ν, S0_mat, λ, λ
    )
    new_state = VMPManualState(new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, nothing, nothing)
    return new_state, rel_change, cache
end

"""
    _vmp_step(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν, S0_mat,
        ::BDMLHierarchicalModel, method::VMPMethod,
    )

Single VMP coordinate-ascent step for the hierarchical model.

Reads ``\\lambda_\\delta = \\mathbb{E}[\tau_\\delta]``, ``\\lambda_\\gamma = \\mathbb{E}[\tau_\\gamma]``
from the current state, calls [`_vmp_step_shared`](@ref), and then updates
``q(\\tau_\\delta)``, ``q(\\tau_\\gamma)`` via closed-form Gamma posterior updates.
"""
function _vmp_step(
        state::VMPManualState, Sxx, xsy, xsd, Sww, ν, S0_mat,
        ::BDMLHierarchicalModel, method::VMPMethod,
    )
    λδ = mean(state.τδ)
    λγ = mean(state.τγ)
    new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, rel_change, cache = _vmp_step_shared(
        state, Sxx, xsy, xsd, Sww, ν, S0_mat, λδ, λγ
    )
    p_dim = length(new_mδ)
    rate_δ = inv(method.bτ) + 0.5 * (tr(new_Vδ) + dot(new_mδ, new_mδ))
    rate_γ = inv(method.bτ) + 0.5 * (tr(new_Vγ) + dot(new_mγ, new_mγ))
    new_τδ = Gamma(method.aτ + p_dim / 2, inv(rate_δ))
    new_τγ = Gamma(method.aτ + p_dim / 2, inv(rate_γ))
    new_state = VMPManualState(new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, new_τδ, new_τγ)
    return new_state, rel_change, cache
end

"""
    _fit_vmp_loop(
        model::M,
        method::VMPMethod{ManualCoordinateAscentVMP},
        state::VMPManualState,
        Sxx, xsy, xsd, Sww, n::Int, ν::Float64, S0_mat::Matrix{Float64};
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    ) where {M <: AbstractBDMLModel}

Run the VMP coordinate-ascent loop until convergence or `n_iterations` is reached.

# Convergence criterion
Relative maximum change:
``\\text{rel\\_change} = \frac{\\max |\theta^{\text{new}} - \theta^{\text{old}}|}{\\max(1, \\max|\theta^{\text{new}}|)}``
where ``\\theta`` collects all parameters ``(m_\\delta, m_\\gamma, V_\\delta, V_\\gamma, S)``.
Convergence is declared when `rel_change <= tolerance` and `iteration > 1`.

# Arguments
- `model`: `BDMLBasicModel` or `BDMLHierarchicalModel`
- `method`: [`VMPMethod`](@ref) with [`ManualCoordinateAscentVMP`](@ref) backend
- `state`: initial [`VMPManualState`](@ref)
- `Sxx`, `xsy`, `xsd`, `Sww`: sufficient statistics
- `n`: number of observations
- `ν`, `S0_mat`: Inverse-Wishart posterior degrees of freedom and prior scale

# Keyword arguments
- `n_iterations::Int = 50`
- `n_draws::Int = 2000`
- `rng::AbstractRNG`
- `show_progress::Bool = false`

# Returns
`(state, converged, actual_iterations, diagnostic_history)`
"""
function _fit_vmp_loop(
    model::M,
    method::VMPMethod{ManualCoordinateAscentVMP},
    state::VMPManualState,
    Sxx, xsy, xsd, Sww, n::Int, ν::Float64, S0_mat::Matrix{Float64};
    n_iterations::Int = 50,
    n_draws::Int = 2000,
    rng::AbstractRNG = Random.default_rng(),
    show_progress::Bool = false,
) where {M <: AbstractBDMLModel}
    tolerance = method.backend.tolerance
    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))

    diagnostic_history = Vector{Float64}(undef, n_iterations)
    converged = false
    actual_iterations = n_iterations

    for iteration in 1:n_iterations
        state, rel_change, cache = _vmp_step(state, Sxx, xsy, xsd, Sww, ν, S0_mat, model, method)
        converged = iteration > 1 && rel_change <= tolerance
        diagnostic_history[iteration] = _vmp_elbo(
            state, Sxx, xsy, xsd, Sww, n, method.ν0, S0_mat, model, method, cache
        )
        show_progress && print("\rVMP manual coordinate ascent: iteration $iteration/$n_iterations")
        if converged
            diagnostic_history[(iteration + 1):end] .= diagnostic_history[iteration]
            actual_iterations = iteration
            break
        end
    end
    show_progress && println()

    return state, converged, actual_iterations, diagnostic_history
end

"""
    _fit_vmp(
        model::BDMLBasicModel,
        method::VMPMethod{ManualCoordinateAscentVMP};
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )

Fit a basic BDML model with conjugate VMP via manual coordinate ascent.

# Algorithm summary
1. Pre-compute sufficient statistics ``S_{xx} = X'X``, ``x_{sy} = X'Y``, etc.
2. Initialise a [`VMPManualState`](@ref) with ``q(\\delta), q(\\gamma) \\sim N(0, 25 I)`` and
   ``q(\\Sigma) \\sim \text{InverseWishart}(\nu_0, S_0)``.
3. Iterate [`_vmp_step`](@ref) / [`_vmp_elbo`](@ref) until convergence.
4. Draw ``n_{\text{draws}}`` posterior samples of ``\\alpha = \\Sigma_{12} / \\Sigma_{22}``
   and rescale to the original data units.

# Keyword arguments
- `n_iterations::Int = 50`: maximum coordinate-ascent iterations
- `n_draws::Int = 2000`: posterior samples for ``\\alpha``
- `rng::AbstractRNG = Random.default_rng()`
- `show_progress::Bool = false`

# Returns
A [`BDMLVMPResult`](@ref) with posterior fields ``(\\delta, \\gamma, \\Sigma)``.

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for
  Causal Inference", arXiv:2508.12688v1, Section 4, Algorithm 1, Section 6.
"""
function _fit_vmp(
    model::BDMLBasicModel,
    method::VMPMethod{ManualCoordinateAscentVMP};
    n_iterations::Int = 50,
    n_draws::Int = 2000,
    rng::AbstractRNG = Random.default_rng(),
    show_progress::Bool = false,
)
    n = nobs(model)
    p = ncovariates(model)
    S0_mat = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    ν = method.ν0 + n

    Sxx = Symmetric(model.X' * model.X)
    xsy = model.X' * model.Y
    xsd = model.X' * model.D
    Sww = Symmetric([dot(model.Y, model.Y) dot(model.Y, model.D); dot(model.Y, model.D) dot(model.D, model.D)])

    state = _initial_state(p, method.ν0, S0_mat, model, method)
    state, converged, actual_iterations, diagnostic_history = _fit_vmp_loop(
        model, method, state, Sxx, xsy, xsd, Sww, n, ν, S0_mat;
        n_iterations = n_iterations,
        n_draws = n_draws,
        rng = rng,
        show_progress = show_progress,
    )

    posterior = (
        δ = MvNormal(state.mδ, Symmetric(state.Vδ)),
        γ = MvNormal(state.mγ, Symmetric(state.Vγ)),
        Σ = state.qΣ,
    )

    α_s_samples = Vector{Float64}(undef, n_draws)
    for s in eachindex(α_s_samples)
        Σ_s = rand(rng, state.qΣ)
        α_s_samples[s] = Σ_s[1, 2] / Σ_s[2, 2]
    end
    α_samples = α_s_samples .* (model.stats.Y_sd / model.stats.D_sd)

    return BDMLVMPResult(
        posterior,
        α_samples,
        α_s_samples,
        model.stats,
        :basic,
        :manual_coordinate_ascent,
        n_iterations,
        actual_iterations,
        diagnostic_history,
        converged,
        diagnostic_history[end],
        :elbo,
    )
end

"""
    _fit_vmp(
        model::BDMLHierarchicalModel,
        method::VMPMethod{ManualCoordinateAscentVMP};
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )

Fit a hierarchical BDML model with conjugate VMP via manual coordinate ascent.

Identical to the basic variant except the prior on coefficients is adaptive:
``\\delta \\sim N(0, \tau_\\delta^{-1} I_p)``, ``\\tau_\\delta \\sim \text{Gamma}(a_\\tau, b_\\tau)``
(and similarly for ``\\gamma``).

The posterior therefore includes ``(\\tau_\\delta, \tau_\\gamma)`` in addition to
``(\\delta, \\gamma, \\Sigma)``.

# Returns
A [`BDMLVMPResult`](@ref) with posterior fields ``(\\delta, \\gamma, \\Sigma, \tau_\\delta, \tau_\\gamma)``.

See also: [`_fit_vmp(::BDMLBasicModel, ...)`](@ref)
"""
function _fit_vmp(
    model::BDMLHierarchicalModel,
    method::VMPMethod{ManualCoordinateAscentVMP};
    n_iterations::Int = 50,
    n_draws::Int = 2000,
    rng::AbstractRNG = Random.default_rng(),
    show_progress::Bool = false,
)
    n = nobs(model)
    p = ncovariates(model)
    S0_mat = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    ν = method.ν0 + n

    Sxx = Symmetric(model.X' * model.X)
    xsy = model.X' * model.Y
    xsd = model.X' * model.D
    Sww = Symmetric([dot(model.Y, model.Y) dot(model.Y, model.D); dot(model.Y, model.D) dot(model.D, model.D)])

    state = _initial_state(p, method.ν0, S0_mat, model, method)
    state, converged, actual_iterations, diagnostic_history = _fit_vmp_loop(
        model, method, state, Sxx, xsy, xsd, Sww, n, ν, S0_mat;
        n_iterations = n_iterations,
        n_draws = n_draws,
        rng = rng,
        show_progress = show_progress,
    )

    posterior = (
        δ = MvNormal(state.mδ, Symmetric(state.Vδ)),
        γ = MvNormal(state.mγ, Symmetric(state.Vγ)),
        Σ = state.qΣ,
        τ_δ = state.τδ,
        τ_γ = state.τγ,
    )

    α_s_samples = Vector{Float64}(undef, n_draws)
    for s in eachindex(α_s_samples)
        Σ_s = rand(rng, state.qΣ)
        α_s_samples[s] = Σ_s[1, 2] / Σ_s[2, 2]
    end
    α_samples = α_s_samples .* (model.stats.Y_sd / model.stats.D_sd)

    return BDMLVMPResult(
        posterior,
        α_samples,
        α_s_samples,
        model.stats,
        :hier,
        :manual_coordinate_ascent,
        n_iterations,
        actual_iterations,
        diagnostic_history,
        converged,
        diagnostic_history[end],
        :elbo,
    )
end
