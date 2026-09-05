# Collapsed VI fitting on the 3-d/5-d marginal posterior

using AdvancedVI
using LogDensityProblems
using Random
using LinearAlgebra
using Optimisers

# Helper to initialize variational distribution for collapsed problem
function _init_collapsed_q(d::Int, fullrank::Bool)
    if fullrank
        L = LowerTriangular(Matrix(Diagonal(fill(0.1, d))))
        return AdvancedVI.FullRankGaussian(zeros(d), L)
    else
        return AdvancedVI.MeanFieldGaussian(zeros(d), Diagonal(fill(0.1, d)))
    end
end

function _configure_collapsed_alg(method::CollapsedVIMethod)
    ad_kwargs = ad_backend_kwargs(method.ad_backend)
    return AdvancedVI.KLMinRepGradDescent(
        method.ad_backend(; ad_kwargs...);
        n_samples = method.n_montecarlo,
        optimizer = Optimisers.Adam(0.005),
        operator = AdvancedVI.ClipScale(),
    )
end

function _fit_collapsed_vi(
        model::AbstractBDMLModel,
        method::CollapsedVIMethod,
        model_type::Symbol;
        n_iterations::Int = 1000,
        n_draws::Int = 2000,
        rng::AbstractRNG = Random.default_rng(),
        show_progress::Bool = true,
    )
    # Build collapsed stats and problem
    c = CollapsedBDML(model)
    collapsed_problem = CollapsedBDMLProblem(c; model_type = model_type)

    d = LogDensityProblems.dimension(collapsed_problem)
    q0 = _init_collapsed_q(d, method.fullrank)
    alg = _configure_collapsed_alg(method)
    configure_ad_backend(method.ad_backend, (;), false)

    @info "Collapsed VI ($(method.ad_backend)): n=$(c.n), p=$(c.p), model_type=$(model_type), dim=$(d), fullrank=$(method.fullrank), iterations=$(n_iterations)"

    q_result, opt_stats, _ = AdvancedVI.optimize(
        rng, alg, n_iterations, collapsed_problem, q0; show_progress = show_progress
    )

    elbo_history = Float64[]
    for stat in opt_stats
        if hasproperty(stat, :elbo)
            push!(elbo_history, stat.elbo)
        end
    end
    final_elbo = isempty(elbo_history) ? -Inf : elbo_history[end]

    converged, conv_msg = check_elbo_convergence(
        elbo_history; min_pct = 0.3, rel_tol = 0.05, check_trend = true, min_iterations = 50
    )
    @info "VI convergence" converged = converged message = conv_msg n_iterations = length(elbo_history)

    vi_samples = rand(rng, q_result, n_draws)
    α_s_samples = Vector{Float64}(undef, n_draws)
    @inbounds for i in 1:n_draws
        sigma_U = 0.1 + exp(vi_samples[1, i])
        sigma_V = 0.1 + exp(vi_samples[2, i])
        rho = tanh(vi_samples[3, i])
        α_s_samples[i] = rho * sigma_U / sigma_V
    end

    scaling_factor = model.stats.Y_sd / model.stats.D_sd
    α_samples = α_s_samples .* scaling_factor

    vi_family_sym = method.fullrank ? :collapsed_fullrank : :collapsed_meanfield

    return BDMLVIResult(
        q_result, α_samples, α_s_samples, model.stats,
        model_type, vi_family_sym, :collapsed, length(elbo_history), elbo_history,
        converged, final_elbo,
    )
end

# Dispatch for basic and hierarchical via model_type
function _fit_impl(model::BDMLBasicModel, method::CollapsedVIMethod; kwargs...)
    return _fit_collapsed_vi(model, method, :basic; kwargs...)
end

function _fit_impl(model::BDMLHierarchicalModel, method::CollapsedVIMethod; kwargs...)
    return _fit_collapsed_vi(model, method, :hier; kwargs...)
end
