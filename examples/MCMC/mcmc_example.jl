### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 7b485e44-306d-4459-92de-3e1464fc54a9
# ╠═╡ show_logs = false
import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))

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
end

# ╔═╡ 1c8a43d6-6328-42d0-874a-b61b2d8fd1ec
md"""
We then define the BDML problem, using the BDML-Hier model
"""

# ╔═╡ cc0e76e1-2932-44de-93af-5f66da7d95e6
prob = BDMLProblem(Y, D, X, model_type=:hier)

# ╔═╡ 7b94053d-4e78-47fe-8a54-e618a97e6221
md"""
We then solve the inference problem via MCMC, using the No-U-Turn Sampler (NUTS), as in the paper:
"""

# ╔═╡ c69bb099-62ab-4ecc-a2f3-77bb3b644a57
sol = fit(
	prob, 
	MCMCMethod(:nuts), 
	n_chains = 4, 
	n_samples=500
)

# ╔═╡ 3ae9455f-70b4-42a1-8a28-42d0a11e0722
summary(sol)

# ╔═╡ Cell order:
# ╠═7b485e44-306d-4459-92de-3e1464fc54a9
# ╠═f0f80cb4-70e6-4b95-bc14-e43d01f5a9e5
# ╟─7eca318e-81c6-413c-8562-8038876024dd
# ╟─6fa27fdd-413f-4841-b49d-a7fd3d513d8f
# ╠═7a7bfba3-6907-4573-99d6-adf7fc9c8799
# ╟─1c8a43d6-6328-42d0-874a-b61b2d8fd1ec
# ╠═cc0e76e1-2932-44de-93af-5f66da7d95e6
# ╟─7b94053d-4e78-47fe-8a54-e618a97e6221
# ╠═c69bb099-62ab-4ecc-a2f3-77bb3b644a57
# ╠═3ae9455f-70b4-42a1-8a28-42d0a11e0722
