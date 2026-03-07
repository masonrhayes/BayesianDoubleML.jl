# Simple VI Implementation (Turing's native vi())
# This is the original VI implementation that uses Turing's built-in ADVI
#
# ADVANTAGES:
# - Uses Turing's native vi() which handles bijectors automatically
# - Works with AutoMooncake (fast after warmup, ~5-10x speedup)
# - Simpler code, less maintenance
# - Mature, battle-tested implementation
#
# DISADVANTAGES:
# - Less control over bijectors
# - No explicit subsampling support (Turing vi() doesn't support it)
# - Hidden complexity in Turing's internals
#
# USE THIS WHEN:
# - You want to use AutoMooncake for maximum performance
# - You can afford warmup iterations (50-100 for rule compilation)
# - You don't need subsampling (n < 10,000)
# - You prefer simplicity over control
#
# vs Unified VI (src/vi/):
# - Unified VI uses AdvancedVI with explicit Bijectors
# - Better for ReverseDiff (stable, no warmup needed)
# - Supports subsampling for large datasets
# - More transparent, easier to debug

using Turing.Variational
using ADTypes

export fit_bdml_vi_simple, bdml_basic_vi, bdml_hier_vi, bdml_basic_vi_rd, bdml_hier_vi_rd

# Include the model definitions
include("model_vi.jl")

# Include the fitting function
include("fit_vi.jl")
