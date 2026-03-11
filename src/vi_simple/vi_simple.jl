# Simple VI Implementation (Turing's native vi())
# This is a simple VI implementation that uses Turing's built-in ADVI
#
# ADVANTAGES:
# - Uses Turing's native vi() which handles bijectors automatically
# - Works with AutoMooncake (fast after warmup, ~5-10x speedup)

using Turing.Variational
using ADTypes

export fit_bdml_vi_simple, bdml_basic_vi, bdml_hier_vi, bdml_basic_vi_rd, bdml_hier_vi_rd

# Include the model definitions
include("model_vi.jl")

# Include the fitting function
include("fit_vi.jl")
