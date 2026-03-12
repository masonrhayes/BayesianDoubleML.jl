# Extended Test Suite for BayesianDoubleML.jl
# Run manually: julia --project=. test/extended/runtests.jl
#
# These tests are not run in CI but can be executed for:
# - Release testing
# - Performance benchmarking
# - AD backend comparisons
# - Large dataset validation
# - Comprehensive validation

using SafeTestsets

println("\n========================================")
println("BayesianDoubleML.jl Extended Test Suite")
println("========================================\n")

# AD Backends - Comprehensive tests with all backends and timing
println("=== Extended AD Backends Tests ===")
println("Testing: AutoReverseDiff, AutoForwardDiff, AutoZygote, AutoMooncake")
println("Includes: Timing comparisons, performance benchmarks\n")

@safetestset "AD Backends Extended" begin
    include("ad_backends_extended.jl")
end

# Subsampling - Large dataset tests (n>10000)
println("\n=== Extended Subsampling Tests ===")
println("Testing: Auto-detection, batch sizes on large datasets")
println("Dataset sizes: n=10000 to 20000\n")

@safetestset "Subsampling Extended" begin
    include("subsampling_extended.jl")
end

println("\n========================================")
println("Extended Tests Complete")
println("========================================")
