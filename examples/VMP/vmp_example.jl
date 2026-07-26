### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 01b90848-8919-11f1-86a3-cf59c8e771e0
# ╠═╡ show_logs = false
begin
    import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))
    Pkg.activate(joinpath(@__DIR__, "../../examples"))
    Pkg.instantiate()
end

# ╔═╡ 82f79d60-4861-43ba-a9c7-d43f434b0afd
begin
    using BayesianDoubleML
    using StableRNGs
    using RxInfer
    using PlutoUI
end

# ╔═╡ 8b469c4c-8e45-40ce-9089-215b72882e3a
md"""
### Data generation
"""

# ╔═╡ 4340088b-9438-49fc-a690-0f07f0e62530
md"""
Let's generate data as per Section 6 [DiTraglia and Liu (2025)](http://arxiv.org/abs/2508.12688).
"""

# ╔═╡ 8ff367ca-753e-4613-835f-09c0184d93f1
begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data as DataFrame
    df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)
end;

# ╔═╡ 5554b354-abc9-4fcd-9e21-0996effeabed
md"""
We then define the BDMLModel, using the hierarchical model (BDML-Hier) from the paper:

As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for $\delta$ and $\gamma$ ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on $\sigma^2_\delta$ and $\sigma^2_\gamma$."
"""

# ╔═╡ 25e871ed-0b54-4e14-bdcd-32195741cba7
model = BDMLModel(df, :y, :d; model_type = :hier)

# ╔═╡ d79788fb-94c0-4fb5-9c68-14144182a92a
md"""
And we then fit the model using Variational Message Passing (VMP). 

In this example, we first try using the manual method using coordinate ascent (`ManualCoordinateAscentVMP()`).

Whether using this method or the RxInfer method, the VMP implementation has the advantage of blazingly fast inference, because it:
- Applies an adjusted model using only conjugate-exponential distributions, simplifying inference. 
- Relies only on the *sufficient statistics* of the data so that the fitting time does not scale with ``n``.

It performs equally well as ADVI but significantly faster; however, it also has the same shortcomings - namely, when ``p`` is large relative to ``n``, the approximation is not typically close to the true causal effect.

"""

# ╔═╡ 473a4905-fece-45ee-b1f6-69f0a73af362
# ╠═╡ show_logs = false
fit!(
    model,
    VMP(; backend = ManualCoordinateAscentVMP()),
    n_iterations = 50,
    n_draws = 2000,
    show_progress = true
)

# ╔═╡ 65ef7695-3d5c-4538-bb2f-ad1063d2052f
md"""
As shown from the results below, VMP's *approximation* to the posterior is not a good approximation for this particular problem. However, in a variety of scenarios (e.g., where the number of observations is relatively large relative to the number of covariates), VMP performs very well and is extremely fast.

"""

# ╔═╡ f098ae2a-89a5-4262-95bf-42584e04a29f
coeftable(model)

# ╔═╡ 25ccb315-787d-4b2b-84b3-3c03ef631d5d
summary(model)

# ╔═╡ 98323223-dc7f-49b2-bc56-62d0591350f0
begin
    n2 = 100_000
    lower_p = floor(sqrt(n2)) |> Int
    upper_p = floor(n2 / 2) |> Int
    default_p = Int(floor(sqrt(n2)))
    @assert lower_p < upper_p
    @bind p2 Slider(lower_p:10:upper_p, show_value = true, default = default_p)
end

# ╔═╡ 8f6392ca-632a-49ab-873f-90678df6c9e8
md"""
## Problems more suitable to VMP

As we see above, for this problem, as with ADVI, VMP is *not* a good fit for the problem above where ``p`` is large relative to ``n``; VMP is not as able to reach a good approximation, at least not with this data generation process. The true causal effect is $(alpha_true), but the above model estimated $(round(coef(model)[1], digits =2)).

However, ADVI is yields a good approximation in a variety of other real-world scenarios; let's try a case where e.g., n=$(n2), p = $(p2).

As a general rule of thumb: in anecdotal testing, variational methods like ADVI and VMP are generally reliable on similar problems when ``n >> p``.
"""

# ╔═╡ c51ca999-58a8-4f5d-94d5-4fef41191c11
# Generate data with more observations
df2 = make_plr_DTL2025(n2, p2, 2.0; alpha = alpha_true, rng = rng);

# ╔═╡ b659cba3-5fe2-4264-8b06-2898b9462795
model2 = BDMLModel(df2, :y, :d; model_type = :hier)

# ╔═╡ 3426efe3-eb2a-42ee-9a6c-069714f852ad
md"""
For example, with n=$(n2) observations and p = $(p2) regressors, VMP gives a good estimate of the posterior in <1 second. 

"""

# ╔═╡ b7951960-fda3-41f3-9fec-c5bd95fe1f5a
@time fit!(
    model2,
    VMP(; backend = ManualCoordinateAscentVMP()),
    n_iterations = 50,
    show_progress = false
);

# ╔═╡ 28763efe-78a1-41a0-9743-d36290c43d73
begin
    summary(model2)
    coeftable(model2)
end

# ╔═╡ a059b9ac-08a2-4ec3-af3d-20429a4a0666
md"""
## RxInfer backend

Instead of the manually-implemented VMP, we can also fit the same model using [RxInfer.jl](https://rxinfer.com/) as the backend. The two deliver nearly identical results, with the manual implementation being slightly more performant.

"""

# ╔═╡ ab5711b0-ec90-4a79-bc2c-95e487a60819
md"""
We could also fit the exact same model using RxInfer as a backend:
"""

# ╔═╡ 405ed58e-0cc6-417a-9c9f-959b1ad619b9
begin
    model2_rx = BDMLModel(df2, :y, :d; model_type = :hier)

    # Fit using RxInfer
    @time fit!(
        model2_rx,
        VMP(; backend = RxInferVMP()),
        n_iterations = 50,
        show_progress = true,
    )

end


# ╔═╡ cecbc982-cb59-4ca1-81b7-ed4491cb387c
begin
    summary(model2_rx)
    coeftable(model2_rx)
end

# ╔═╡ Cell order:
# ╟─01b90848-8919-11f1-86a3-cf59c8e771e0
# ╠═82f79d60-4861-43ba-a9c7-d43f434b0afd
# ╟─8b469c4c-8e45-40ce-9089-215b72882e3a
# ╟─4340088b-9438-49fc-a690-0f07f0e62530
# ╠═8ff367ca-753e-4613-835f-09c0184d93f1
# ╟─5554b354-abc9-4fcd-9e21-0996effeabed
# ╠═25e871ed-0b54-4e14-bdcd-32195741cba7
# ╟─d79788fb-94c0-4fb5-9c68-14144182a92a
# ╠═473a4905-fece-45ee-b1f6-69f0a73af362
# ╟─65ef7695-3d5c-4538-bb2f-ad1063d2052f
# ╠═f098ae2a-89a5-4262-95bf-42584e04a29f
# ╠═25ccb315-787d-4b2b-84b3-3c03ef631d5d
# ╟─8f6392ca-632a-49ab-873f-90678df6c9e8
# ╠═c51ca999-58a8-4f5d-94d5-4fef41191c11
# ╠═b659cba3-5fe2-4264-8b06-2898b9462795
# ╠═98323223-dc7f-49b2-bc56-62d0591350f0
# ╟─3426efe3-eb2a-42ee-9a6c-069714f852ad
# ╠═b7951960-fda3-41f3-9fec-c5bd95fe1f5a
# ╠═28763efe-78a1-41a0-9743-d36290c43d73
# ╠═a059b9ac-08a2-4ec3-af3d-20429a4a0666
# ╟─ab5711b0-ec90-4a79-bc2c-95e487a60819
# ╠═405ed58e-0cc6-417a-9c9f-959b1ad619b9
# ╠═cecbc982-cb59-4ca1-81b7-ed4491cb387c
