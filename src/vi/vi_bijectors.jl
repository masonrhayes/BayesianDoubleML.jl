# Explicit Bijectors for Unified VI Implementation
# Transforms parameters from unconstrained space (ℝ^d) to constrained space

using Bijectors
using Distributions
using LinearAlgebra

export bijector

"""
    Bijectors.bijector(model::BDMLVIModel)

Create a bijector that transforms parameters from unconstrained space 
(where VI optimization happens) to constrained space (where model is defined).

This follows the ADVI pattern: we optimize in unconstrained space (real numbers),
and the bijector transforms parameters to constrained space before computing 
log-posterior.

# Transformation Details

## Hierarchical Model (model_type=:hier)
Parameter order: [log_σ²_δ, log_σ²_γ, θ_Y..., θ_D..., log_σ_U, log_σ_V, logit_ρ_raw]

After transformation: [σ²_δ, σ²_γ, θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
- Log-normal → exp() → positive values (σ²_δ, σ²_γ, σ_U, σ_V)
- Identity → real values (θ_Y, θ_D)
- Logistic → [0,1] (ρ_raw)

## Basic Model (model_type=:basic)
Parameter order: [θ_Y..., θ_D..., log_σ_U, log_σ_V, logit_ρ_raw]

After transformation: [θ_Y..., θ_D..., σ_U, σ_V, ρ_raw]
- Identity → real values (θ_Y, θ_D)
- Log-normal → exp() → positive values (σ_U, σ_V)
- Logistic → [0,1] (ρ_raw)

# Returns
`Bijectors.Stacked`: A stacked bijector that applies different transformations
to different parameter ranges.

# Examples
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic)
b = bijector(model)  # Returns unconstrained → constrained bijector

# Transform unconstrained parameters to constrained
θ_unconstrained = zeros(2*p + 3)  # All zeros
θ_constrained = Bijectors.transform(b, θ_unconstrained)
# θ_constrained now has: θ_Y=0, θ_D=0, σ_U=1, σ_V=1, ρ_raw=0.5
```

# Implementation Notes
The bijector uses the inverse of the standard distribution bijectors:
- bijector(LogNormal) = log (constrained → unconstrained)
- inverse(bijector(LogNormal)) = exp (unconstrained → constrained) ✓

This is the correct direction for VI: unconstrained optimization → constrained model.

# See Also
- `LogDensityProblems.logdensity(::BDMLVIModel, θ)`: Uses this bijector internally
- `AdvancedVI.optimize`: Optimizes in unconstrained space
"""
function Bijectors.bijector(model::BDMLVIModel)
    n, p = size(model.X)
    T = model.T

    if model.model_type == :hier
        # Hierarchical model parameter ranges:
        # [log_σ²_δ, log_σ²_γ, θ_Y(p), θ_D(p), log_σ_U, log_σ_V, logit_ρ_raw]
        ranges = [
            1:1,                    # log_σ²_δ → σ²_δ (positive via exp)
            2:2,                    # log_σ²_γ → σ²_γ (positive via exp)
            3:(2 + p),                  # θ_Y (already real, identity)
            (3 + p):(2 + 2p),              # θ_D (already real, identity)
            (3 + 2p):(3 + 2p),             # log_σ_U → σ_U (positive via exp)
            (4 + 2p):(4 + 2p),             # log_σ_V → σ_V (positive via exp)
            (5 + 2p):(5 + 2p),              # logit_ρ_raw → ρ_raw (0,1 via logistic)
        ]

        # Distributions define the target space (constrained)
        # We need the inverse bijector: unconstrained → constrained
        dists = [
            LogNormal(T(0), T(1)),      # σ²_δ: exp(unconstrained) → positive
            LogNormal(T(0), T(1)),        # σ²_γ: exp(unconstrained) → positive
            MvNormal(zeros(T, p), I),  # θ_Y: identity (already real)
            MvNormal(zeros(T, p), I),  # θ_D: identity (already real)
            LogNormal(T(0), T(1)),        # σ_U: exp(unconstrained) → positive
            LogNormal(T(0), T(1)),        # σ_V: exp(unconstrained) → positive
            Beta(T(2), T(2)),              # ρ_raw: logistic(unconstrained) → (0,1)
        ]
    else
        # Basic model parameter ranges:
        # [θ_Y(p), θ_D(p), log_σ_U, log_σ_V, logit_ρ_raw]
        ranges = [
            1:p,                    # θ_Y (already real, identity)
            (p + 1):2p,                 # θ_D (already real, identity)
            (2p + 1):(2p + 1),             # log_σ_U → σ_U (positive via exp)
            (2p + 2):(2p + 2),             # log_σ_V → σ_V (positive via exp)
            (2p + 3):(2p + 3),              # logit_ρ_raw → ρ_raw (0,1 via logistic)
        ]

        dists = [
            MvNormal(zeros(T, p), I),  # θ_Y: identity (already real)
            MvNormal(zeros(T, p), I),  # θ_D: identity (already real)
            LogNormal(T(0), T(1)),        # σ_U: exp(unconstrained) → positive
            LogNormal(T(0), T(1)),        # σ_V: exp(unconstrained) → positive
            Beta(T(2), T(2)),              # ρ_raw: logistic(unconstrained) → (0,1)
        ]
    end

    # bijector(d) returns: constrained → unconstrained (e.g., exp → log)
    # We need the inverse: unconstrained → constrained (e.g., log → exp)
    # This gives us: optimize in ℝ^d, transform to model space for likelihood
    b_fwd = Bijectors.bijector.(dists)  # constrained → unconstrained
    b_inv = inverse.(b_fwd)              # unconstrained → constrained ✓

    return Bijectors.Stacked(b_inv, ranges)
end

"""
    inverse_bijector(model::BDMLVIModel)

Get the inverse bijector: constrained → unconstrained.

This is occasionally useful for transforming initial values or debugging.

# Returns
`Bijectors.Stacked`: constrained → unconstrained bijector

# Examples
```julia
model = BDMLVIModel(Y, D, X; model_type=:basic)
b_inv = inverse_bijector(model)

# Transform constrained initial guess to unconstrained space
θ_constrained = [zeros(p); zeros(p); 1.0; 1.0; 0.5]  # σ_U=1, σ_V=1, ρ_raw=0.5
θ_unconstrained = Bijectors.transform(b_inv, θ_constrained)
```
"""
function inverse_bijector(model::BDMLVIModel)
    b = bijector(model)
    return inverse(b)
end
