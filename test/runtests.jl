# BayesianDoubleML.jl Test Suite
# Entry point with SafeTestsets for proper test isolation
#
# Each @safetestset runs in its own module to prevent:
# - Variable name collisions
# - Side effects between tests
# - Pollution of global namespace
#
# Follows SciML Style Guide: https://github.com/SciML/SciMLStyle

using SafeTestsets

# Phase 1: Core functionality tests (fast, no inference)
# These tests verify basic types, constructors, and utilities
println("\n=== Phase 1: Core Functionality ===")
@time @safetestset "Core Types and Utilities" begin
    include("core.jl")
end
@time @safetestset "BDMLProblem Constructors" begin
    include("problems.jl")
end
@time @safetestset "Inference Methods" begin
    include("methods.jl")
end

# Phase 2: Inference method tests (slower, involves sampling)
# These tests verify MCMC and VI inference work correctly
println("\n=== Phase 2: Inference Methods ===")
@time @safetestset "MCMC Inference" begin
    include("mcmc.jl")
end
@time @safetestset "Unified VI" begin
    include("vi_unified.jl")
end
@time @safetestset "Simple VI" begin
    include("vi_simple.jl")
end

# Phase 3: Feature tests (testing specific features)
# These tests verify AD backends, subsampling, diagnostics
println("\n=== Phase 3: Feature Tests ===")
@time @safetestset "AD Backends" begin
    include("ad_backends.jl")
end
@time @safetestset "Subsampling" begin
    include("subsampling.jl")
end
@time @safetestset "Diagnostics" begin
    include("diagnostics.jl")
end
@time @safetestset "Alpha Extraction" begin
    include("alpha_extraction.jl")
end

# Phase 4: Integration tests (slowest, real data)
# These tests verify real data handling
println("\n=== Phase 4: Integration Tests ===")
@time @safetestset "Real Data (MCMC)" begin
    include("real_data_mcmc.jl")
end
@time @safetestset "Real Data (VI)" begin
    include("real_data_vi.jl")
end

println("\n=== All Tests Complete ===")
