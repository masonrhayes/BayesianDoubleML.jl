# Conjugate VMP via manual coordinate ascent (structured joint q(delta,gamma)).
# Joint variational family q(delta,gamma) retained via eigenbasis of X'X.
# See DESIGN.md §3a and prototype/structured_vmp.jl

"""
    VMPManualState{T,V<:AbstractVector{T},M<:AbstractMatrix{T},QΣ,Qτ}

Parametric state struct for the conjugate VMP coordinate-ascent algorithm
with structured joint `q(delta,gamma)`.

# Type parameters
- `T`: scalar numeric type (typically `Float64`)
- `V`: concrete vector type for coefficient means / diagonal variances in eigenbasis
- `M`: concrete matrix type for the eigenvector matrix `Q`
- `QΣ`: concrete type of the variational posterior for the `2×2` error covariance `Σ`
- `Qτ`: concrete type of the variational posteriors for precision hyperparameters `τ_δ` and `τ_γ`.
  `Nothing` for the basic model; `Gamma{Float64}` for the hierarchical model.

# Fields
- `m1::V`: variational mean of `δ` in eigenbasis (`mδ = Q*m1`)
- `m2::V`: variational mean of `γ` in eigenbasis (`mγ = Q*m2`)
- `v11::V`: diagonal of `Cov(delta,delta)` in eigenbasis (`Vδ = Q*diag(v11)*Q'`)
- `v12::V`: diagonal of `Cov(delta,gamma)` in eigenbasis (`Cov = Q*diag(v12)*Q'`)
- `v22::V`: diagonal of `Cov(gamma,gamma)` in eigenbasis
- `Q::M`: eigenvectors of `X'X`
- `lam::V`: eigenvalues of `X'X`
- `qΣ::QΣ`: variational posterior `q(Sigma)` as `InverseWishart`
- `τδ::Qτ`: variational posterior `q(τ_δ)` (or `nothing` for basic)
- `τγ::Qτ`: variational posterior `q(τ_γ)` (or `nothing` for basic)

# Notes
For a `BDMLBasicModel` `Qτ = Nothing`; for a `BDMLHierarchicalModel` `Qτ = Gamma{Float64}`.
Retains `Cov(delta,gamma)` that the mean-field `q(delta)q(gamma)` drops.
"""
struct VMPManualState{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, QΣ, Qτ}
    m1::V
    m2::V
    v11::V
    v12::V
    v22::V
    Q::M
    lam::V
    qΣ::QΣ
    τδ::Qτ
    τγ::Qτ
end

# Helpers to recover original-coordinate moments for reporting
_state_mδ(s::VMPManualState) = s.Q * s.m1
_state_mγ(s::VMPManualState) = s.Q * s.m2
function _state_Vδ(s::VMPManualState)
    return Symmetric(s.Q * Diagonal(s.v11) * s.Q')
end
function _state_Vγ(s::VMPManualState)
    return Symmetric(s.Q * Diagonal(s.v22) * s.Q')
end

"""
Return the effective residual degrees of freedom consumed by the joint
coefficient block. Each eigen-direction contributes half the trace of its
multivariate ridge hat matrix, and therefore contributes one in the unpenalized
full-rank limit.
"""
function _vmp_effective_df(lam, qΣ::InverseWishart, λδ::Real, λγ::Real)
    ν, S = Distributions.params(qΣ)
    Ω = ν * inv(Symmetric(S))
    Ω11, Ω12, Ω22 = Ω[1, 1], Ω[1, 2], Ω[2, 2]
    effective_df = 0.0

    @inbounds for eigenvalue in lam
        eigenvalue = max(eigenvalue, zero(eigenvalue))
        a = eigenvalue * Ω11 + λδ
        c = eigenvalue * Ω12
        d = eigenvalue * Ω22 + λγ
        determinant = a * d - c * c
        v11 = d / determinant
        v12 = -c / determinant
        v22 = a / determinant
        effective_df += eigenvalue * (
            Ω11 * v11 + 2 * Ω12 * v12 + Ω22 * v22
        ) / 2
    end
    return effective_df
end

"""
Reduce the Inverse-Wishart degrees of freedom by `effective_df`, scaling its
scale matrix so that the covariance mean is unchanged. This calibrates alpha
uncertainty using effective residual degrees of freedom without changing the
VMP coordinate updates. It is exact for the alpha marginal in the flat
coefficient-prior limit; under shrinkage it is an approximation.
"""
function _vmp_adjust_covariance(qΣ::InverseWishart, effective_df::Real)
    ν, S = Distributions.params(qΣ)
    d = size(S, 1)
    corrected_ν = ν - effective_df
    corrected_ν > d + 1 || throw(ArgumentError("effective residual degrees of freedom must exceed $(d + 1)"))

    # Preserve E[Σ]. A scalar change to S does not affect Σ₁₂ / Σ₂₂.
    corrected_S = S .* ((corrected_ν - d - 1) / (ν - d - 1))
    return InverseWishart(corrected_ν, corrected_S)
end

function _draw_vmp_alpha_samples(rng::AbstractRNG, qΣ::InverseWishart, n_draws::Int)
    α = Vector{Float64}(undef, n_draws)
    for i in eachindex(α)
        Σ = rand(rng, qΣ)
        α[i] = Σ[1, 2] / Σ[2, 2]
    end
    return α
end

"""
    _logmultigamma(a::Real, d::Int)

Multivariate log-gamma function `logγ_d(a)` for a `d`-dimensional Wishart / InverseWishart.
"""
function _logmultigamma(a::Real, d::Int)
    return d * (d - 1) / 4 * log(pi) + sum(
        Distributions.loggamma(a + (1 - j) / 2) for j in 1:d
    )
end

"""
    _expected_logdet(qΣ::InverseWishart)

Expected log-determinant `E_{q(Sigma)}[log|Sigma|]` under an `InverseWishart` distribution.
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
"""
function _inversewishart_entropy(qΣ::InverseWishart)
    ν, S = Distributions.params(qΣ)
    d = size(S, 1)
    elogdet = _expected_logdet(qΣ)
    return -ν / 2 * logdet(Symmetric(S)) + ν * d / 2 * log(2) +
        _logmultigamma(ν / 2, d) + (ν + d + 1) / 2 * elogdet + ν * d / 2
end

"""
    _vmp_elbo_core(state::VMPManualState, R::Matrix{Float64}, logdet_joint::Float64, n::Int, ν0::Float64, S0_mat::Matrix{Float64})

Shared ELBO terms identical for basic and hierarchical models, now with joint entropy.

Computes:
1. Data log-likelihood `-n d/2 log2π - n/2 E[log|Sigma|] - tr(Omega R)/2`
2. Inverse-Wishart prior on `Sigma`
3. Joint entropy `H[q(delta,gamma)] = 0.5*(2p*(1+log2π)+logdet_joint)` plus `H[q(Sigma)]`
"""
function _vmp_elbo_core(
        state::VMPManualState, R::Matrix{Float64}, logdet_joint::Float64, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64},
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

    p = length(state.m1)
    elbo += (2p * (1 + log(2 * pi)) + logdet_joint) / 2
    elbo += _inversewishart_entropy(state.qΣ)
    return elbo
end

"""
    _vmp_elbo(state, G, Sww, n, ν0, S0_mat, ::BDMLBasicModel, ::VMPMethod, cache)

ELBO for the basic BDML model with fixed Gaussian priors `delta, gamma ~ N(0, 25 I_p)`.
"""
function _vmp_elbo(
        state::VMPManualState, G::Matrix{Float64}, Sww::AbstractMatrix, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLBasicModel, ::VMPMethod, cache,
    )
    R, logdet_joint = cache.R, cache.logdet_joint
    elbo = _vmp_elbo_core(state, R, logdet_joint, n, ν0, S0_mat)
    # Q = ||m||^2 + tr(V) = dot(m1,m1)+sum(v11) (orthogonal Q preserves norm/trace)
    Qδ = cache.t11
    Qγ = cache.t22
    p = length(state.m1)
    elbo += -p / 2 * log(2 * pi * 25.0) - Qδ / 50
    elbo += -p / 2 * log(2 * pi * 25.0) - Qγ / 50
    return elbo
end

"""
    _vmp_elbo(state, G, Sww, n, ν0, S0_mat, ::BDMLHierarchicalModel, method::VMPMethod, cache)

ELBO for the hierarchical BDML model with adaptive shrinkage.
"""
function _vmp_elbo(
        state::VMPManualState, G::Matrix{Float64}, Sww::AbstractMatrix, n::Int,
        ν0::Float64, S0_mat::Matrix{Float64}, ::BDMLHierarchicalModel, method::VMPMethod, cache,
    )
    R, logdet_joint = cache.R, cache.logdet_joint
    elbo = _vmp_elbo_core(state, R, logdet_joint, n, ν0, S0_mat)
    Qδ = cache.t11
    Qγ = cache.t22
    p = length(state.m1)
    for (qτ, Q) in ((state.τδ, Qδ), (state.τγ, Qγ))
        A, θ = Distributions.params(qτ)
        Eτ = A * θ
        Elogτ = Distributions.digamma(A) + log(θ)
        elbo += -p / 2 * log(2 * pi) + p / 2 * Elogτ - Eτ * Q / 2
        elbo += (method.aτ - 1) * Elogτ - Eτ / method.bτ -
            Distributions.loggamma(method.aτ) - method.aτ * log(method.bτ)
        elbo += A + log(θ) + Distributions.loggamma(A) + (1 - A) * Distributions.digamma(A)
    end
    return elbo
end

"""
    _initial_state(p, ν0, S0_mat, Q, lam, ::BDMLBasicModel, ::VMPMethod)

Create initial state for basic model in eigenbasis: `m1=m2=0`, `v11=v22=25`, `v12=0`.
"""
function _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, Q::Matrix{Float64}, lam::Vector{Float64}, ::BDMLBasicModel, ::VMPMethod)
    m1 = zeros(p)
    m2 = zeros(p)
    v11 = fill(25.0, p)
    v22 = fill(25.0, p)
    v12 = zeros(p)
    qΣ = InverseWishart(ν0, S0_mat)
    return VMPManualState(m1, m2, v11, v12, v22, Q, lam, qΣ, nothing, nothing)
end

"""
    _initial_state(p, ν0, S0_mat, Q, lam, ::BDMLHierarchicalModel, method::VMPMethod)

Create initial state for hierarchical model (also `q(τ)`).
"""
function _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, Q::Matrix{Float64}, lam::Vector{Float64}, ::BDMLHierarchicalModel, method::VMPMethod)
    m1 = zeros(p)
    m2 = zeros(p)
    v11 = fill(25.0, p)
    v22 = fill(25.0, p)
    v12 = zeros(p)
    qΣ = InverseWishart(ν0, S0_mat)
    τδ = Gamma(method.aτ, method.bτ)
    τγ = Gamma(method.aτ, method.bτ)
    return VMPManualState(m1, m2, v11, v12, v22, Q, lam, qΣ, τδ, τγ)
end

"""
    _vmp_step_shared(state, G, Sww, ν, S0_mat, λδ, λγ)

Execute one structured coordinate-ascent VMP step: `q(delta,gamma)` jointly.

In eigenbasis `X'X = Q Lam Q'`, the `2p×2p` precision becomes `p` independent `2×2`
blocks `M_j = lam_j*Omega + diag(λδ,λγ)`. Invert each `2×2` analytically, keeping
`Cov(delta,gamma)` via `i12 = -c/det`.

Returns `(new_m1,new_m2,new_v11,new_v12,new_v22,new_qΣ,rel_change,cache)` where
`cache` holds `R`, `logdet_joint`, and `t11/t22` for the ELBO.
"""
function _vmp_step_shared(
        state::VMPManualState, G::Matrix{Float64}, Sww::AbstractMatrix, ν::Float64, S0_mat::Matrix{Float64},
        λδ::Float64, λγ::Float64,
    )
    ν_Σ, Ψ_Σ = Distributions.params(state.qΣ)
    Ω = ν_Σ * inv(Symmetric(Ψ_Σ))
    Ω = Matrix(Symmetric(Ω))
    Om11, Om12, Om22 = Ω[1, 1], Ω[1, 2], Ω[2, 2]
    GOm = G * Ω  # p×2, (Q'X'W)*Omega

    p = length(state.m1)
    new_m1 = Vector{Float64}(undef, p)
    new_m2 = Vector{Float64}(undef, p)
    new_v11 = Vector{Float64}(undef, p)
    new_v22 = Vector{Float64}(undef, p)
    new_v12 = Vector{Float64}(undef, p)

    s11 = 0.0; s22 = 0.0; s12 = 0.0
    t11 = 0.0; t22 = 0.0
    logdet_prec = 0.0

    @inbounds for j in 1:p
        lamj = state.lam[j]
        a = lamj * Om11 + λδ
        c = lamj * Om12
        d = lamj * Om22 + λγ
        det2 = a * d - c * c
        # numerical guard: M_j is PD by construction, det2>0
        if det2 <= 1.0e-14
            # jitter if near-singular (should not happen with positive λ)
            det2 = max(det2, 1.0e-12)
        end
        i11 = d / det2
        i22 = a / det2
        i12 = -c / det2
        new_v11[j] = i11
        new_v22[j] = i22
        new_v12[j] = i12
        # mean: m_j = inv(M_j) * (GOm_j)
        g1 = GOm[j, 1]; g2 = GOm[j, 2]
        x1 = i11 * g1 + i12 * g2
        x2 = i12 * g1 + i22 * g2
        new_m1[j] = x1
        new_m2[j] = x2
        s11 += lamj * (x1 * x1 + i11)
        s22 += lamj * (x2 * x2 + i22)
        s12 += lamj * (x1 * x2 + i12)
        t11 += x1 * x1 + i11
        t22 += x2 * x2 + i22
        logdet_prec += log(det2)
    end
    logdet_joint = -logdet_prec  # logdet of joint covariance = - sum logdet(M_j)

    r11 = Sww[1, 1] - 2 * dot(G[:, 1], new_m1) + s11
    r22 = Sww[2, 2] - 2 * dot(G[:, 2], new_m2) + s22
    r12 = Sww[1, 2] - dot(G[:, 1], new_m2) - dot(G[:, 2], new_m1) + s12
    S = S0_mat + [r11 r12; r12 r22]
    S = Matrix(Symmetric(S))
    isposdef(Symmetric(S)) || throw(ArgumentError("VMP covariance update is not positive definite"))
    new_qΣ = InverseWishart(ν, S)

    change = max(
        maximum(abs, new_m1 .- state.m1),
        maximum(abs, new_m2 .- state.m2),
        maximum(abs, new_v11 .- state.v11),
        maximum(abs, new_v22 .- state.v22),
        maximum(abs, new_v12 .- state.v12),
        maximum(abs, S .- Ψ_Σ),
    )
    scale = max(1.0, maximum(abs, new_m1), maximum(abs, new_m2), maximum(abs, S))
    rel_change = change / scale

    R = Matrix(Symmetric([r11 r12; r12 r22]))
    cache = (R = R, logdet_joint = logdet_joint, t11 = t11, t22 = t22)

    return new_m1, new_m2, new_v11, new_v12, new_v22, new_qΣ, rel_change, cache
end

"""
    _vmp_step(state, G, Sww, ν, S0_mat, ::BDMLBasicModel, ::VMPMethod)

Single step for basic model (fixed λ=1/25).
"""
function _vmp_step(
        state::VMPManualState, G::Matrix{Float64}, Sww::AbstractMatrix, ν, S0_mat,
        ::BDMLBasicModel, ::VMPMethod,
    )
    λ = 1.0 / 25.0
    new_m1, new_m2, new_v11, new_v12, new_v22, new_qΣ, rel_change, cache = _vmp_step_shared(
        state, G, Sww, ν, S0_mat, λ, λ
    )
    new_state = VMPManualState(new_m1, new_m2, new_v11, new_v12, new_v22, state.Q, state.lam, new_qΣ, nothing, nothing)
    return new_state, rel_change, cache
end

"""
    _vmp_step(state, G, Sww, ν, S0_mat, ::BDMLHierarchicalModel, method::VMPMethod)

Single step for hierarchical model (λ = E[τ]).
"""
function _vmp_step(
        state::VMPManualState, G::Matrix{Float64}, Sww::AbstractMatrix, ν, S0_mat,
        ::BDMLHierarchicalModel, method::VMPMethod,
    )
    λδ = mean(state.τδ)
    λγ = mean(state.τγ)
    new_m1, new_m2, new_v11, new_v12, new_v22, new_qΣ, rel_change, cache = _vmp_step_shared(
        state, G, Sww, ν, S0_mat, λδ, λγ
    )
    p_dim = length(new_m1)
    rate_δ = inv(method.bτ) + 0.5 * cache.t11
    rate_γ = inv(method.bτ) + 0.5 * cache.t22
    new_τδ = Gamma(method.aτ + p_dim / 2, inv(rate_δ))
    new_τγ = Gamma(method.aτ + p_dim / 2, inv(rate_γ))
    new_state = VMPManualState(new_m1, new_m2, new_v11, new_v12, new_v22, state.Q, state.lam, new_qΣ, new_τδ, new_τγ)
    return new_state, rel_change, cache
end

"""
    _fit_vmp_loop(model, method, state, G, Sww, n, ν, S0_mat; n_iterations, ...)

Run coordinate ascent until convergence or `n_iterations`.
"""
function _fit_vmp_loop(
        model::M,
        method::VMPMethod{ManualCoordinateAscentVMP},
        state::VMPManualState,
        G::Matrix{Float64}, Sww::AbstractMatrix, n::Int, ν::Float64, S0_mat::Matrix{Float64};
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
        state, rel_change, cache = _vmp_step(state, G, Sww, ν, S0_mat, model, method)
        converged = iteration > 1 && rel_change <= tolerance
        diagnostic_history[iteration] = _vmp_elbo(
            state, G, Sww, n, method.ν0, S0_mat, model, method, cache
        )
        show_progress && print("\rVMP manual coordinate ascent (structured): iteration $iteration/$n_iterations")
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
    _fit_vmp(model::BDMLBasicModel, method::VMPMethod{ManualCoordinateAscentVMP}; ...)

Fit basic BDML with structured VMP (eigenbasis, joint q(delta,gamma)).
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

    # One-time eigenbasis of X'X: O(p^3) once, O(p) per sweep thereafter
    F = eigen(Symmetric(model.X' * model.X))
    lam = F.values
    # Guard against tiny negative eigenvalues from numerical error
    @inbounds for i in eachindex(lam)
        if lam[i] < 0 && lam[i] > -1.0e-10
            lam[i] = 0.0
        end
    end
    Q = F.vectors
    G = Q' * (model.X' * hcat(model.Y, model.D))  # p×2
    Sww = Symmetric([dot(model.Y, model.Y) dot(model.Y, model.D); dot(model.Y, model.D) dot(model.D, model.D)])

    state = _initial_state(p, method.ν0, S0_mat, Q, lam, model, method)
    state, converged, actual_iterations, diagnostic_history = _fit_vmp_loop(
        model, method, state, G, Sww, n, ν, S0_mat;
        n_iterations = n_iterations,
        n_draws = n_draws,
        rng = rng,
        show_progress = show_progress,
    )

    mδ = _state_mδ(state)
    mγ = _state_mγ(state)
    Vδ = _state_Vδ(state)
    Vγ = _state_Vγ(state)
    effective_df = _vmp_effective_df(state.lam, state.qΣ, 1 / 25, 1 / 25)
    corrected_qΣ = _vmp_adjust_covariance(state.qΣ, effective_df)
    posterior = (
        δ = MvNormal(mδ, Vδ),
        γ = MvNormal(mγ, Vγ),
        Σ = corrected_qΣ,
        Σ_vmp = state.qΣ,
        effective_df = effective_df,
    )

    α_s_samples = _draw_vmp_alpha_samples(rng, corrected_qΣ, n_draws)
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
    _fit_vmp(model::BDMLHierarchicalModel, method::VMPMethod{ManualCoordinateAscentVMP}; ...)

Fit hierarchical BDML with structured VMP.
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

    F = eigen(Symmetric(model.X' * model.X))
    lam = F.values
    @inbounds for i in eachindex(lam)
        if lam[i] < 0 && lam[i] > -1.0e-10
            lam[i] = 0.0
        end
    end
    Q = F.vectors
    G = Q' * (model.X' * hcat(model.Y, model.D))
    Sww = Symmetric([dot(model.Y, model.Y) dot(model.Y, model.D); dot(model.Y, model.D) dot(model.D, model.D)])

    state = _initial_state(p, method.ν0, S0_mat, Q, lam, model, method)
    state, converged, actual_iterations, diagnostic_history = _fit_vmp_loop(
        model, method, state, G, Sww, n, ν, S0_mat;
        n_iterations = n_iterations,
        n_draws = n_draws,
        rng = rng,
        show_progress = show_progress,
    )

    mδ = _state_mδ(state)
    mγ = _state_mγ(state)
    Vδ = _state_Vδ(state)
    Vγ = _state_Vγ(state)
    effective_df = _vmp_effective_df(
        state.lam, state.qΣ, mean(state.τδ), mean(state.τγ)
    )
    corrected_qΣ = _vmp_adjust_covariance(state.qΣ, effective_df)
    posterior = (
        δ = MvNormal(mδ, Vδ),
        γ = MvNormal(mγ, Vγ),
        Σ = corrected_qΣ,
        Σ_vmp = state.qΣ,
        effective_df = effective_df,
        τ_δ = state.τδ,
        τ_γ = state.τγ,
    )

    α_s_samples = _draw_vmp_alpha_samples(rng, corrected_qΣ, n_draws)
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
