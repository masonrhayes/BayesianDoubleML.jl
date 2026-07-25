# Conjugate VMP via manual coordinate ascent (sufficient-statistics form).

abstract type AbstractVMPPriorPolicy end

struct FixedPriorPolicy <: AbstractVMPPriorPolicy end

struct HierarchicalPriorPolicy <: AbstractVMPPriorPolicy
    aτ::Float64
    bτ::Float64
end

get_precision(::FixedPriorPolicy, ::Any) = 1.0 / 25.0
get_precision(p::HierarchicalPriorPolicy, τ::Gamma) = mean(τ)

update_τ(::FixedPriorPolicy, m, V, p) = nothing
function update_τ(p::HierarchicalPriorPolicy, m::AbstractVector, V::AbstractMatrix, p_dim::Int)
    rate = inv(p.bτ) + 0.5 * (tr(V) + dot(m, m))
    return Gamma(p.aτ + p_dim / 2, inv(rate))
end

struct VMPManualState
    mδ::Vector{Float64}
    mγ::Vector{Float64}
    Vδ::Matrix{Float64}
    Vγ::Matrix{Float64}
    qΣ::InverseWishart
    τδ::Union{Nothing, Gamma}
    τγ::Union{Nothing, Gamma}
end

function _initial_state(p::Int, ν0::Float64, S0_mat::Matrix{Float64}, policy::AbstractVMPPriorPolicy)
    mδ = zeros(p)
    mγ = zeros(p)
    Vδ = Matrix{Float64}(I, p, p) * 25.0
    Vγ = copy(Vδ)
    qΣ = InverseWishart(ν0, S0_mat)
    τδ = policy isa HierarchicalPriorPolicy ? Gamma(policy.aτ, policy.bτ) : nothing
    τγ = policy isa HierarchicalPriorPolicy ? Gamma(policy.aτ, policy.bτ) : nothing
    return VMPManualState(mδ, mγ, Vδ, Vγ, qΣ, τδ, τγ)
end

function _vmp_step(state::VMPManualState, Sxx, xsy, xsd, Sww, ν, S0_mat, policy::AbstractVMPPriorPolicy)
    ν_Σ, Ψ_Σ = Distributions.params(state.qΣ)
    Ω = ν_Σ * inv(Symmetric(Ψ_Σ))
    Ω = Matrix(Symmetric(Ω))
    λδ = get_precision(policy, state.τδ)
    λγ = get_precision(policy, state.τγ)

    # Coefficient updates via Cholesky (parallel mean-field, matching RxInfer)
    Pδ = Symmetric(Ω[1, 1] .* Sxx + λδ .* I)
    Pγ = Symmetric(Ω[2, 2] .* Sxx + λγ .* I)
    hδ = Ω[1, 1] .* xsy + Ω[1, 2] .* (xsd - Sxx * state.mγ)
    hγ = Ω[2, 2] .* xsd + Ω[1, 2] .* (xsy - Sxx * state.mδ)

    Fδ = cholesky(Pδ)
    Fγ = cholesky(Pγ)
    new_mδ = Fδ \ hδ
    new_mγ = Fγ \ hγ
    new_Vδ = Matrix(inv(Fδ))
    new_Vγ = Matrix(inv(Fγ))

    # Expected residual cross-products, including coefficient uncertainty
    Aδ = new_Vδ + new_mδ * new_mδ'
    Aγ = new_Vγ + new_mγ * new_mγ'
    r11 = Sww[1, 1] - 2 * dot(xsy, new_mδ) + dot(Sxx, Aδ)
    r22 = Sww[2, 2] - 2 * dot(xsd, new_mγ) + dot(Sxx, Aγ)
    r12 = Sww[1, 2] - dot(xsy, new_mγ) - dot(xsd, new_mδ) + dot(new_mδ, Sxx * new_mγ)
    S = S0_mat + [r11 r12; r12 r22]
    S = Matrix(Symmetric(S))
    isposdef(Symmetric(S)) || throw(ArgumentError("VMP covariance update is not positive definite"))
    new_qΣ = InverseWishart(ν, S)

    new_τδ = update_τ(policy, new_mδ, new_Vδ, length(new_mδ))
    new_τγ = update_τ(policy, new_mγ, new_Vγ, length(new_mγ))

    change = max(
        maximum(abs, new_mδ - state.mδ), maximum(abs, new_mγ - state.mγ),
        maximum(abs, new_Vδ - state.Vδ), maximum(abs, new_Vγ - state.Vγ),
        maximum(abs, S - Ψ_Σ)
    )
    scale = max(1.0, maximum(abs, new_mδ), maximum(abs, new_mγ), maximum(abs, S))
    rel_change = change / scale

    new_state = VMPManualState(new_mδ, new_mγ, new_Vδ, new_Vγ, new_qΣ, new_τδ, new_τγ)
    return new_state, rel_change
end

function _fit_vmp(
        ::ManualCoordinateAscentVMP,
        model::AbstractBDMLModel,
        method::VMPMethod;
        n_iterations::Int = 50,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = false,
    )
    n = nobs(model)
    p = ncovariates(model)
    hierarchical = model isa BDMLHierarchicalModel
    policy = hierarchical ? HierarchicalPriorPolicy(method.aτ, method.bτ) : FixedPriorPolicy()
    tolerance = method.backend.tolerance

    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))

    S0_mat = method.S0 === nothing ? Matrix{Float64}(I, 2, 2) : method.S0
    ν = method.ν0 + n

    # Sufficient statistics
    Y = model.Y
    D = model.D
    X = model.X
    Sxx = Symmetric(X'X)
    xsy = X'Y
    xsd = X'D
    Sww = Symmetric([dot(Y, Y) dot(Y, D); dot(Y, D) dot(D, D)])

    state = _initial_state(p, method.ν0, S0_mat, policy)
    diagnostic_history = Vector{Float64}(undef, n_iterations)
    converged = false
    actual_iterations = n_iterations

    for iteration in 1:n_iterations
        state, rel_change = _vmp_step(state, Sxx, xsy, xsd, Sww, ν, S0_mat, policy)
        converged = iteration > 1 && rel_change <= tolerance
        diagnostic_history[iteration] = rel_change
        show_progress && print("\rVMP manual coordinate ascent: iteration $iteration/$n_iterations")
        if converged
            diagnostic_history[(iteration + 1):end] .= diagnostic_history[iteration]
            actual_iterations = iteration
            break
        end
    end
    show_progress && println()

    posterior = hierarchical ?
        (
            δ = MvNormal(state.mδ, Symmetric(state.Vδ)),
            γ = MvNormal(state.mγ, Symmetric(state.Vγ)),
            Σ = state.qΣ,
            τ_δ = state.τδ,
            τ_γ = state.τγ,
        ) :
        (
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
        hierarchical ? :hier : :basic,
        :manual_coordinate_ascent,
        n_iterations,
        actual_iterations,
        diagnostic_history,
        converged,
        diagnostic_history[end],
        :parameter_change,
    )
end
