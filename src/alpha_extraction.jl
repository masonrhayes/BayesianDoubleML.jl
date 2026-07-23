# Additional alpha extraction for VI matrix samples
# Used internally by the dispatch system

"""
    extract_alpha(vi_samples::Matrix{Float64}, p::Int, model_type::Symbol)

Extract α samples from VI posterior samples matrix using paper's Equation 15.

# Mathematical Derivation (DiTraglia & Liu 2025, Section 4)

From the bivariate reduced form model (Equations 13-15):
    Y = X'δ + U,  where U = ε + αV   (Eq. 12)
    D = X'γ + V                      (Eq. 5)

The causal effect α is recovered from error covariance:
    α = Cov(U, V) / Var(V) = σ_UV / σ²_V   (Eq. 15)

For VI models, the correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1].

Therefore:
    α = ρ * σ_U / σ_V
      = (2*ρ_raw - 1) * σ_U / σ_V

# Arguments
- `vi_samples::Matrix{Float64}`: Samples from variational posterior (parameters × samples)
- `p::Int`: Number of control variables (covariates)
- `model_type::Symbol`: :basic or :hier

# Returns
- `Vector{Float64}`: Alpha samples (α = ρ·σ_U / σ_V)

# Parameter Indexing

For hierarchical models:
- [σ²_δ, σ²_γ, δ(1:p), γ(1:p), σ_U, σ_V, ρ_raw]
- Indices: σ²_δ=1, σ²_γ=2, δ=3:(2+p), γ=(3+p):(2+2p), σ_U=(3+2p), σ_V=(4+2p), ρ_raw=(5+2p)

For basic models:
- [δ(1:p), γ(1:p), σ_U, σ_V, ρ_raw]
- Indices: δ=1:p, γ=(p+1):2p, σ_U=(2p+1), σ_V=(2p+2), ρ_raw=(2p+3)

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Equation 15.
"""
function extract_alpha(vi_samples::Matrix{Float64}, p::Int, model_type::Symbol)
    n_samples = size(vi_samples, 2)
    α_samples = Vector{Float64}(undef, n_samples)

    # Calculate parameter indices
    if model_type == :hier
        # Hierarchical: [σ²_δ, σ²_γ, δ(1:p), γ(1:p), σ_U, σ_V, ρ_raw]
        σ_U_idx = 3 + 2 * p
        σ_V_idx = 4 + 2 * p
        ρ_raw_idx = 5 + 2 * p
    else
        # Basic: [δ(1:p), γ(1:p), σ_U, σ_V, ρ_raw]
        σ_U_idx = 2 * p + 1
        σ_V_idx = 2 * p + 2
        ρ_raw_idx = 2 * p + 3
    end

    # Verify indices
    d = size(vi_samples, 1)
    @assert σ_U_idx <= d "σ_U index exceeds dimension"
    @assert σ_V_idx <= d "σ_V index exceeds dimension"
    @assert ρ_raw_idx <= d "ρ_raw index exceeds dimension"

    for i in 1:n_samples
        σ_U = vi_samples[σ_U_idx, i]
        σ_V = vi_samples[σ_V_idx, i]
        ρ_raw = vi_samples[ρ_raw_idx, i]

        # Transform ρ from [0, 1] to [-1, 1] using paper's parameterization
        ρ = 2 * ρ_raw - 1

        # Compute alpha using Equation 15: α = ρ·σ_U / σ_V
        α_samples[i] = ρ * σ_U / σ_V
    end

    return α_samples
end

"""
    extract_alpha(vnt_samples::AbstractVector{<:VarNamedTuple})

Extract α samples from constrained `VarNamedTuple` posterior draws, as returned by
`rand(::Turing.Variational.VIResult, n)` (Turing 0.46+ Simple VI path).

The VI models parameterize the correlation as ρ_raw ~ Beta(2, 2) on [0, 1], so
α = (2*ρ_raw - 1) * σ_U / σ_V (Equation 15).
"""
function extract_alpha(vnt_samples::AbstractVector{<:VarNamedTuple})
    return Float64[
        (2 * vnt[@varname(ρ_raw)] - 1) * vnt[@varname(σ_U)] / vnt[@varname(σ_V)] for
        vnt in vnt_samples
    ]
end
