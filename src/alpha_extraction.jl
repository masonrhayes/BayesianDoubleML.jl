# Additional alpha extraction for VI matrix samples
# Used internally by the dispatch system

"""
    extract_alpha(vi_samples::Matrix{Float64}, p::Int, model_type::Symbol)

Extract α samples from VI posterior samples matrix.

For VI models, the correlation is parameterized as ρ_raw ~ Beta(2, 2) on [0, 1],
then transformed to ρ = 2*ρ_raw - 1 on [-1, 1].

The samples are in CONSTRAINED space (already transformed by bijectors).

# Arguments
- `vi_samples::Matrix{Float64}`: Samples from variational posterior (parameters × samples)
- `p::Int`: Number of control variables (covariates)
- `model_type::Symbol`: :basic or :hier

# Returns
- `Vector{Float64}`: Alpha samples (α = ρ * σ_U / σ_V)
"""
function extract_alpha(vi_samples::Matrix{Float64}, p::Int, model_type::Symbol)
    n_samples = size(vi_samples, 2)
    α_samples = Vector{Float64}(undef, n_samples)

    # Calculate parameter indices
    if model_type == :hier
        # Hierarchical: [σ²_δ, σ²_γ, θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
        σ_U_idx = 3 + 2 * p
        σ_V_idx = 4 + 2 * p
        ρ_raw_idx = 5 + 2 * p
    else
        # Basic: [θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
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

        # Transform ρ from [0, 1] to [-1, 1]
        ρ = 2 * ρ_raw - 1

        # Compute alpha
        α_samples[i] = ρ * σ_U / σ_V
    end

    return α_samples
end
