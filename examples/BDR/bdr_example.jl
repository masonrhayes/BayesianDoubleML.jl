### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 0106d4c8-a946-11f1-87ef-f1f6fd32361b
begin
    import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))
    Pkg.activate(joinpath(@__DIR__, "../../examples"))
    Pkg.instantiate()
end

# ╔═╡ c53fc165-bda1-4385-a8bb-f26cbb08ade2
begin
    using BayesianDoubleML
    using StableRNGs
    using PlutoUI
    using CairoMakie, DataFrames
end

# ╔═╡ a882d74e-c502-475a-86b4-b0e17696fe27
begin
    # Generate the continuous-treatment exposure-response simulation design from Antonelli, Papadogeorgou, and Dominici (2022), Section 5.2.
    SEED = 33

    rng = StableRNG(SEED)

    # Generate data as DataFrame
    df = make_er_APD2022(StableRNG(SEED))
end;

# ╔═╡ a9ae387a-b7d0-48f9-9a94-177cfdcee5c4
begin
    # Set up the model

    model = BayesDRModel(df, :y, :d, treatment_type = :auto)

    # Fit the model
    fit!(
        model,
        n_samples = 2000,
        n_burn = 500,
        n_boot = 1000,
        thin = 2,
        n_chains = 1;
        rng = StableRNG(SEED)
    )

    summary(model)

end

# ╔═╡ 6c2ff869-7c83-4c47-a7f1-8df065efaa7d
fig = plot_exposure_response_curve(model)

# ╔═╡ Cell order:
# ╟─0106d4c8-a946-11f1-87ef-f1f6fd32361b
# ╠═c53fc165-bda1-4385-a8bb-f26cbb08ade2
# ╠═a882d74e-c502-475a-86b4-b0e17696fe27
# ╠═a9ae387a-b7d0-48f9-9a94-177cfdcee5c4
# ╟─af1f1da9-6d11-4c12-83d2-3144cf03b4eb
# ╠═6c2ff869-7c83-4c47-a7f1-8df065efaa7d
