# Multiple Dispatch-Based Fitting Interface for BDML
#
# This file provides a unified `fit!()` interface that mutates the model
# and stores results internally. The model serves as both specification
# and result container.
#
# The actual implementation dispatches on both:
# 1. Model type (BDMLBasicModel vs BDMLHierarchicalModel)
# 2. Inference method (MCMCMethod, CollapsedVIMethod, or VMPMethod)

export fit!

"""
    fit!(model::AbstractBDMLModel, method::AbstractInferenceMethod; force=false, kwargs...)

Fit a BDML model using the specified inference method, storing results in the model.

This is a mutating function that modifies the model in-place. After fitting,
results can be extracted using `coeftable()`, `extract_alpha()`, `summary()`, etc.

If the model has already been fitted, a warning is shown unless `force=true`
is passed to allow refitting.

# Arguments
- `model::AbstractBDMLModel`: The model to fit (data + metadata)
- `method::AbstractInferenceMethod`: The inference algorithm to use

# Keyword Arguments
- `force::Bool=false`: Allow refitting an already-fitted model

For MCMC methods (including `CollapsedMCMC`):
- `n_samples::Int=2000`: Number of posterior samples to draw
- `n_chains::Int=4`: Number of MCMC chains to run

For VI methods:
- `n_iterations::Int=1000`: Number of optimization iterations
- `n_draws::Int=2000`: Number of posterior samples to draw after fitting

# Returns
`nothing` (follows standard Julia mutating function convention)

# Examples
```julia
# Create model
model = BDMLModel(Y, D, X; model_type=:basic)

# Fit with MCMC (NUTS)
fit!(model, MCMCMethod(:nuts); n_samples=2000, n_chains=4)

# Fit with collapsed MCMC (NUTS on the exact marginal posterior)
fit!(model, CollapsedMCMC(); n_samples=2000, n_chains=4)

# Fit with CollapsedVI
fit!(model, CollapsedVI(); n_iterations=1000)

# Extract results
coef_table = coeftable(model)
summary(model)

# Refit with force (changes the stored result)
fit!(model, VMP(); force=true)
```

# Multiple Dispatch
The actual implementation dispatches on both model and method:
- `_fit_impl(::BDMLBasicModel, ::MCMCMethod)` - Basic model MCMC
- `_fit_impl(::BDMLHierarchicalModel, ::MCMCMethod)` - Hierarchical MCMC
- `_fit_impl(::AbstractBDMLModel, ::CollapsedMCMCMethod)` - Collapsed MCMC
- `_fit_impl(::BDMLBasicModel, ::CollapsedVIMethod)` - Basic model CollapsedVI
- `_fit_impl(::BDMLHierarchicalModel, ::CollapsedVIMethod)` - Hierarchical CollapsedVI
- `_fit_impl(::BDMLBasicModel, ::VMPMethod)` - Basic conjugate VMP (RxInfer extension)
- `_fit_impl(::BDMLHierarchicalModel, ::VMPMethod)` - Hierarchical conjugate VMP (RxInfer extension)

See also: [`BDMLModel`](@ref), [`MCMCMethod`](@ref), [`CollapsedMCMCMethod`](@ref), [`CollapsedVIMethod`](@ref), [`isfitted`](@ref)
"""
function fit!(
        model::AbstractBDMLModel, method::AbstractInferenceMethod;
        force::Bool = false, kwargs...
    )
    # Check if already fitted
    if model.is_fitted && !force
        @warn "Model has already been fitted. Use force=true to refit."
        return nothing
    end

    # Dispatch to implementation based on model type and method
    result = _fit_impl(model, method; kwargs...)

    # Store result in model
    model.result = result
    model.is_fitted = true
    model.last_method = method

    return nothing
end

"""
    fit!(model::AbstractBDMLModel; force=false, kwargs...)

Fit a model using default method (MCMC with NUTS).

Convenience method that defaults to NUTS sampler.
"""
function fit!(model::AbstractBDMLModel; force::Bool = false, kwargs...)
    return fit!(model, MCMCMethod(:nuts); force = force, kwargs...)
end

# Error fallback for unimplemented combinations
function _fit_impl(model::AbstractBDMLModel, method::AbstractInferenceMethod; kwargs...)
    error("No fit implementation defined for model type $(typeof(model)) with method $(typeof(method))")
end

# VMP routing via backend dispatch
function _fit_impl(model::AbstractBDMLModel, method::VMPMethod; kwargs...)
    return _fit_vmp(model, method; kwargs...)
end

# Fallback when a backend extension is not loaded
function _fit_vmp(model::AbstractBDMLModel, method::VMPMethod; kwargs...)
    return error(
        "VMP backend $(typeof(method.backend)) is not available. " *
            "Load the required extension (e.g., `using RxInfer`) and try again."
    )
end

# MCMC dispatch

function _fit_mcmc(
        model::AbstractBDMLModel, method::MCMCMethod, turing_model;
        n_samples::Int = 2000, n_chains::Int = 4, implementation::Symbol = :full,
    )
    implementation in (:full, :collapsed) || throw(
        ArgumentError("implementation must be :full or :collapsed, got $(repr(implementation))"),
    )

    sampler = NUTS(method.target_acceptance; max_depth = method.max_depth, adtype = AutoForwardDiff())
    chain = if n_chains == 1
        sample(turing_model, sampler, n_samples; progress = true)
    else
        sample(turing_model, sampler, MCMCThreads(), n_samples, n_chains; progress = true)
    end

    α_s_samples = extract_alpha(chain)
    α_samples = α_s_samples .* (model.stats.Y_sd / model.stats.D_sd)
    return BDMLMCMCResult(chain, α_samples, α_s_samples, model.stats, model_type(model), implementation)
end

"""
    _fit_impl(model::BDMLBasicModel, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit basic BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization (Turing's native approach).
"""
function _fit_impl(
        model::BDMLBasicModel, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )
    return _fit_mcmc(model, method, bdml_basic(model.Y, model.D, model.X); n_samples, n_chains)
end

"""
    _fit_impl(model::BDMLHierarchicalModel, method::MCMCMethod; n_samples=2000, n_chains=4)

Fit hierarchical BDML model using MCMC (NUTS or HMC).

Uses LKJCholesky correlation parameterization.
"""
function _fit_impl(
        model::BDMLHierarchicalModel, method::MCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4
    )
    return _fit_mcmc(model, method, bdml_hier(model.Y, model.D, model.X); n_samples, n_chains)
end

"""
    _fit_impl(model::AbstractBDMLModel, method::CollapsedMCMCMethod; n_samples=2000, n_chains=4)

Fit a BDML model with NUTS on the exact marginal posterior obtained by
integrating out `(δ, γ)`. This is equivalent to the corresponding full MCMC
model but samples only the error-scale, correlation, and (for `:hier`)
variance parameters.
"""
function _fit_impl(
        model::AbstractBDMLModel, method::CollapsedMCMCMethod;
        n_samples::Int = 2000, n_chains::Int = 4,
    )
    method.sampler.algorithm == :nuts || error(
        "Collapsed MCMC supports only NUTS; got $(method.sampler.algorithm)",
    )

    collapsed_model = bdml_collapsed(CollapsedBDML(model))
    return _fit_mcmc(
        model, method.sampler, collapsed_model;
        n_samples, n_chains, implementation = :collapsed,
    )
end
