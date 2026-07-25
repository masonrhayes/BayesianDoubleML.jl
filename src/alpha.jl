"""
    extract_alpha(chain::FlexiChains.VNChain)

Extract ``\\alpha`` samples from MCMC chain using the paper's transformation ``(Eq. 15)``.

# Mathematical Derivation (DiTraglia & Liu 2025)

The BDML model uses a bivariate reduced form parameterization (Section 4):

**Structural equation:**
    ``Y = \\alpha D + X'\\beta + \\epsilon``,    where ``\\epsilon \\perp V`` (Eq. 6)

**Reduced form:**
    ``Y = X'\\delta + U``,  where ``U = \\epsilon + \\alpha V``   (Eq. 12)
    ``D = X'\\gamma + V``                      (Eq. 5)

Since ``\\epsilon`` is uncorrelated with V by assumption, the covariance between the 
reduced form errors is:
    ``\\text{Cov}(U, V) = \\text{Cov}(\\epsilon + \\alpha V, V) = \\alpha \\cdot \\text{Var}(V)``

Therefore, the causal effect ``\\alpha`` can be recovered from the error covariance:
    ``\\alpha = \\text{Cov}(U, V) / \\text{Var}(V) = \\sigma_{UV} / \\sigma^2_V``   (Eq. 15)

Using the correlation parameterization ``\\sigma_{UV} = \\rho \\cdot \\sigma_U \\cdot \\sigma_V``:
    ``\\alpha = \\rho \\cdot \\sigma_U \\cdot \\sigma_V / \\sigma^2_V = \\rho \\cdot \\sigma_U / \\sigma_V``

# Implementation

For MCMC with LKJCholesky, the chain stores the Cholesky factor L where:
    ``\\rho = L[2,1] / L[1,1]``

Since ``L[1,1] = 1`` for correlation matrices, ``\\rho = L[2,1]``.

This function extracts ``\\sigma_U``, ``\\sigma_V``, and ``\\rho`` from the MCMC chain, then computes:
    ``\\alpha = \\rho \\cdot \\sigma_U / \\sigma_V``

# Arguments
- `chain::FlexiChains.VNChain`: MCMC chain containing posterior samples

# Returns
- `Vector{Float64}`: Posterior samples of ``\\alpha``

# References
- DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for
  Causal Inference", arXiv:2508.12688v1, Section 4, Equation 15.
"""
function extract_alpha(chain::FlexiChains.VNChain)
    σ_U_samples = vec(Array(chain[:σ_U]))
    σ_V_samples = vec(Array(chain[:σ_V]))

    # Extract correlation from LKJCholesky factor via structure-preserving indexing
    R_chol_samples = vec(Array(chain[:R_chol]))
    ρ_samples = map(c -> c.L[2, 1] / c.L[1, 1], R_chol_samples)

    # Compute alpha using Equation 15: α = ρ * σ_U / σ_V
    α_samples = @. ρ_samples * σ_U_samples / σ_V_samples

    return α_samples
end

function extract_alpha(result::BDMLMCMCResult)
    return result.alpha_samples
end

"""
    extract_alpha(result::BDMLVIResult)

Extract ``\alpha`` samples from a BDMLVIResult.
"""
function extract_alpha(result::BDMLVIResult)
    return result.alpha_samples
end

"""
    extract_alpha(result::BDMLVMPResult)

Extract ``\alpha`` samples from a BDMLVMPResult.
"""
function extract_alpha(result::BDMLVMPResult)
    return result.alpha_samples
end
