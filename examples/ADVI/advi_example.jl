### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ 206fed88-1b2a-11f1-891f-7fd268f9692b
# ╠═╡ show_logs = false
begin
    import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))
    Pkg.activate(joinpath(@__DIR__, "../../examples"))
    Pkg.instantiate()
end

# ╔═╡ a62b95cc-3877-4c62-be36-17cad34734a1
begin
    using BayesianDoubleML
    using Mooncake # explictly import Mooncake for AD
    using StableRNGs
    using PlutoUI
end

# ╔═╡ f95bc5b1-62a4-49a4-bc98-0e8ebf749a19
md"""
# BayesianDoubleML VI example
"""

# ╔═╡ 668d4810-707d-4fad-ad0e-de354df5c382
md"""
### Data generation 
"""

# ╔═╡ b26f4d74-e791-4435-860c-c0ebfb9b6dbc
md"""
Let's generate data as per Section 6 [DiTraglia and Liu (2025)](http://arxiv.org/abs/2508.12688).
"""

# ╔═╡ b600a47b-d9d5-4552-b673-6eda27f04871
begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data as DataFrame
    df = make_plr_DTL2025(n, p, 2.0; alpha = alpha_true, rng = rng)
end;

# ╔═╡ eacec63a-2d16-4f52-ac22-c8ba2add8a03
md"""
### Model setup 
"""

# ╔═╡ ebf5e3c0-3e95-45d1-a734-7177f14a0182
md"""
We then define the BDMLModel, using the hierarchical model (BDML-Hier) from the paper:

As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for $\delta$ and $\gamma$ ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on $\sigma^2_\delta$ and $\sigma^2_\gamma$."
"""

# ╔═╡ a9fa5ba9-f25d-4cf0-b9bb-99101c6f90af
model = BDMLModel(df, :y, :d; model_type = :hier)

# ╔═╡ 3e5ab29c-0961-4c34-83dc-2c8ca295fef8
md"""
### Model fitting 
"""

# ╔═╡ cad72553-b33b-445d-85f2-28bec0d2a20b
md"""
And we then fit the model using Automatic Differentiation Variational Inference (ADVI). 

In this example, we first try using the SimpleVIMethod with the AutoMooncake AD backend. (Note: AutoMooncake from [Mooncake.jl](https://chalk-lab.github.io/Mooncake.jl/stable/) provides extremely fast automatic differentiation, at the cost of a longer compile time.)
"""

# ╔═╡ 751ef964-a74b-4a41-a9b9-799241bebda0
fit!(
    model,
    SimpleVIMethod(; ad_backend = AutoMooncake),
    n_iterations = 1_000,
    show_progress = false
);

# ╔═╡ 1251af4f-8941-425a-bef4-0bbb999e420f
begin
    summary(model)
    coeftable(model)
end

# ╔═╡ 9a46ba39-21f4-4c32-8e55-a60cf253aab7
begin
    n2 = 1_000
    lower_p = floor(sqrt(n2)) |> Int
    upper_p = floor(n2 / 2) |> Int
    default_p = Int(floor(sqrt(n2)))
    @assert lower_p < upper_p
    @bind p2 Slider(lower_p:10:upper_p, show_value = true, default = default_p)
end

# ╔═╡ cf1670f6-97c5-4130-a7f3-8bf7e2e9b066
md"""
## Problems more suitable to ADVI

As we see above, for this problem, ADVI is *not* a good fit for the problem above where ``p`` is large relative to ``n``; ADVI is not as able to reach a good approximation, at least not with this data generation process. The true causal effect is $(alpha_true), but the above model estimated $(round(coef(model)[1], digits =2)).

However, ADVI is yields a good approximation in a variety of other real-world scenarios; let's try a case where e.g., n=$(n2), p = $(p2).

As a general rule of thumb: in anecdotal testing, ADVI methods are generally reliable on similar problems when ``n >> p``.
"""

# ╔═╡ a452de95-aa14-472b-8be9-50e18ecab69d
begin
    # Generate data as DataFrame
    df2 = make_plr_DTL2025(n2, p2, 2.0; alpha = alpha_true, rng = rng)
end;

# ╔═╡ f3c45acf-78a2-44d0-87fc-6eb472ae7d77
model2 = BDMLModel(df2, :y, :d; model_type = :hier)

# ╔═╡ 66ca6f4b-d0ae-4481-bf13-dba9f983c3fe
fit!(
    model2,
    SimpleVIMethod(; ad_backend = AutoMooncake),
    n_iterations = 1_000,
    show_progress = false
);

# ╔═╡ d3d7184c-094b-4a4e-b0f8-f16382e2ec76
begin
    summary(model2)
    coeftable(model2)
end

# ╔═╡ Cell order:
# ╟─206fed88-1b2a-11f1-891f-7fd268f9692b
# ╟─f95bc5b1-62a4-49a4-bc98-0e8ebf749a19
# ╠═a62b95cc-3877-4c62-be36-17cad34734a1
# ╟─668d4810-707d-4fad-ad0e-de354df5c382
# ╟─b26f4d74-e791-4435-860c-c0ebfb9b6dbc
# ╠═b600a47b-d9d5-4552-b673-6eda27f04871
# ╟─eacec63a-2d16-4f52-ac22-c8ba2add8a03
# ╠═ebf5e3c0-3e95-45d1-a734-7177f14a0182
# ╠═a9fa5ba9-f25d-4cf0-b9bb-99101c6f90af
# ╟─3e5ab29c-0961-4c34-83dc-2c8ca295fef8
# ╠═cad72553-b33b-445d-85f2-28bec0d2a20b
# ╠═751ef964-a74b-4a41-a9b9-799241bebda0
# ╠═1251af4f-8941-425a-bef4-0bbb999e420f
# ╟─cf1670f6-97c5-4130-a7f3-8bf7e2e9b066
# ╠═9a46ba39-21f4-4c32-8e55-a60cf253aab7
# ╠═a452de95-aa14-472b-8be9-50e18ecab69d
# ╠═f3c45acf-78a2-44d0-87fc-6eb472ae7d77
# ╠═66ca6f4b-d0ae-4481-bf13-dba9f983c3fe
# ╠═d3d7184c-094b-4a4e-b0f8-f16382e2ec76
