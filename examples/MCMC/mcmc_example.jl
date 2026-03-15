### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 7b485e44-306d-4459-92de-3e1464fc54a9
# ╠═╡ show_logs = false
import Pkg; Pkg.develop(path = "../..")

# ╔═╡ b0422dcb-72f8-4689-8b60-4ce6af4456c6
# ╠═╡ show_logs = false
Pkg.activate("..")

# ╔═╡ f0f80cb4-70e6-4b95-bc14-e43d01f5a9e5
# ╠═╡ show_logs = false
begin
    using BayesianDoubleML
    using StableRNGs
end

# ╔═╡ 7eca318e-81c6-413c-8562-8038876024dd
md"""
# BayesianDoubleML MCMC example
"""

# ╔═╡ d0feb7d0-bde7-40da-8e78-6dcbfaadb859
md"""
### Data generation
"""

# ╔═╡ 6fa27fdd-413f-4841-b49d-a7fd3d513d8f
md"""
Let's generate data as per Section 6 of the paper 
"""

# ╔═╡ 7a7bfba3-6907-4573-99d6-adf7fc9c8799
begin
    # Define parameters
    n = 200
    p = 100
    alpha_true = 2.0

    rng = StableRNG(42)

    # Generate data
    Y, D, X = generate_dgp_table1(n, p, 2.0; alpha_true = alpha_true, rng = rng)
end;

# ╔═╡ 85d4a20b-812f-493a-bc53-1cd1ae750234
md"""
### Model setup
"""

# ╔═╡ 1c8a43d6-6328-42d0-874a-b61b2d8fd1ec
md"""
We then define the `BDMLModel`, using the BDML-Hier model. 

As per Section 6 of the paper, the BDML-Hier model "allows different standard deviations in the normal shrinkage priors for $\delta$ and $\gamma$ ... with a hierarchical prior that places independent Inverse-Gamma(2, 2) hyper-priors on $\sigma^2_\delta$ and $\sigma^2_\gamma$."
"""

# ╔═╡ cc0e76e1-2932-44de-93af-5f66da7d95e6
model = BDMLModel(Y, D, X, model_type = :hier)

# ╔═╡ aa8bc797-3255-4f85-83a5-24adf7e3c0f9
md"""
### Model fitting
"""

# ╔═╡ 7b94053d-4e78-47fe-8a54-e618a97e6221
md"""
We then solve the inference problem via MCMC, using the No-U-Turn Sampler (NUTS), as in the paper:
"""

# ╔═╡ c69bb099-62ab-4ecc-a2f3-77bb3b644a57
fit!(
    model,
    MCMCMethod(:nuts),
    n_chains = 8,
    n_samples = 400
)

# ╔═╡ 6b773f27-4581-4b60-a1fb-5a815770e18a
md"""
### Model results
"""

# ╔═╡ 3ae9455f-70b4-42a1-8a28-42d0a11e0722
begin 
	summary(model);
	coeftable(model)
end

# ╔═╡ Cell order:
# ╟─7b485e44-306d-4459-92de-3e1464fc54a9
# ╟─b0422dcb-72f8-4689-8b60-4ce6af4456c6
# ╟─7eca318e-81c6-413c-8562-8038876024dd
# ╠═f0f80cb4-70e6-4b95-bc14-e43d01f5a9e5
# ╟─d0feb7d0-bde7-40da-8e78-6dcbfaadb859
# ╟─6fa27fdd-413f-4841-b49d-a7fd3d513d8f
# ╠═7a7bfba3-6907-4573-99d6-adf7fc9c8799
# ╟─85d4a20b-812f-493a-bc53-1cd1ae750234
# ╟─1c8a43d6-6328-42d0-874a-b61b2d8fd1ec
# ╠═cc0e76e1-2932-44de-93af-5f66da7d95e6
# ╟─aa8bc797-3255-4f85-83a5-24adf7e3c0f9
# ╟─7b94053d-4e78-47fe-8a54-e618a97e6221
# ╠═c69bb099-62ab-4ecc-a2f3-77bb3b644a57
# ╟─6b773f27-4581-4b60-a1fb-5a815770e18a
# ╠═3ae9455f-70b4-42a1-8a28-42d0a11e0722
