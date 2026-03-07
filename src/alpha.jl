function extract_alpha(chain::MCMCChains.Chains)
    σ_U_samples = vec(Array(chain[:σ_U]))
    σ_V_samples = vec(Array(chain[:σ_V]))

    # Extract correlation from LKJCholesky factor
    # The chain stores R_chol.L components as R_chol.L[1, 1], R_chol.L[2, 1], etc.
    # Note: The variable names have spaces in them!

    n_samples = length(σ_U_samples)

    # Get L[1,1] and L[2,1] - note the spaces in the variable names!
    L_11 = vec(Array(chain[Symbol("R_chol.L[1, 1]")]))
    L_21 = vec(Array(chain[Symbol("R_chol.L[2, 1]")]))

    # ρ = L[2,1] / L[1,1] (for correlation matrices, L[1,1] = 1)
    ρ_samples = L_21 ./ L_11

    # Compute alpha: α = ρ * σ_U / σ_V
    α_samples = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        α_samples[i] = ρ_samples[i] * σ_U_samples[i] / σ_V_samples[i]
    end

    return α_samples
end

function extract_alpha(result::BDMLResult)
    return result.alpha_samples
end

"""
    extract_alpha(result::BDMLVIResult)

Extract α samples from a BDMLVIResult.
"""
function extract_alpha(result::BDMLVIResult)
    return result.alpha_samples
end
