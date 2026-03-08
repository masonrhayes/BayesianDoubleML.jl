"""
    extract_alpha(chain::MCMCChains.Chains)

Extract α samples from MCMC chain using the paper's transformation (Equation 15).

# Mathematical Derivation (DiTraglia & Liu 2025)

The BDML model uses a bivariate reduced form parameterization (Section 4):

**Structural equation:**
    Y = αD + X'β + ε,    where ε ⊥ V (Eq. 6)

**Reduced form:**
    Y = X'δ + U,  where U = ε + αV   (Eq. 12)
    D = X'γ + V                      (Eq. 5)

Since ε is uncorrelated with V by assumption, the covariance between the 
reduced form errors is:
    Cov(U, V) = Cov(ε + αV, V) = α·Var(V)

Therefore, the causal effect α can be recovered from the error covariance:
    α = Cov(U, V) / Var(V) = σ_UV / σ²_V   (Eq. 15)

Using the correlation parameterization σ_UV = ρ·σ_U·σ_V:
    α = ρ·σ_U·σ_V / σ²_V = ρ·σ_U / σ_V

# Implementation

For MCMC with LKJCholesky, the chain stores the Cholesky factor L where:
    ρ = L[2,1] / L[1,1]

Since L[1,1] = 1 for correlation matrices, ρ = L[2,1].

This function extracts σ_U, σ_V, and ρ from the MCMC chain, then computes:
    α = ρ * σ_U / σ_V

# Arguments
- `chain::MCMCChains.Chains`: MCMC chain containing posterior samples

# Returns
- `Vector{Float64}`: Posterior samples of α

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
  Causal Inference", arXiv:2508.12688v1, Section 4, Equation 15.
"""
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

    # Compute alpha using Equation 15: α = ρ * σ_U / σ_V
    α_samples = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        α_samples[i] = ρ_samples[i] * σ_U_samples[i] / σ_V_samples[i]
    end

    return α_samples
end

function extract_alpha(result::BDMLMCMCResult)
    return result.alpha_samples
end

"""
    extract_alpha(result::BDMLVIResult)

Extract α samples from a BDMLVIResult.
"""
function extract_alpha(result::BDMLVIResult)
    return result.alpha_samples
end
