# BayesianDoubleML.jl Test Suite
# Entry point with SafeTestsets for proper test isolation
# Optimized for CI - minimal smoke tests, no timing comparisons
#
using SafeTestsets

# Explicitly import AD backends for DifferentiationInterface compatibility
import Enzyme
import Mooncake

# Phase 1: Core functionality tests (fast, no inference)
# These tests verify basic types, constructors, and utilities
println("\n=== Phase 1: Core Functionality ===")

@safetestset "Core Types and Utilities" begin
    include("core.jl")
end

@safetestset "BDMLModel Constructors" begin
    include("models.jl")
end

@safetestset "Inference Methods" begin
    include("methods.jl")
end

# Phase 2: Inference method tests (slower, involves sampling)
# These tests verify MCMC and VI inference work correctly
println("\n=== Phase 2: Inference Methods ===")

@safetestset "MCMC Inference" begin
    include("mcmc.jl")
end
@safetestset "VI (Unified and Simple)" begin
    include("vi.jl")
end

# Phase 3: Feature tests (smoke tests only for CI)
# Extended tests available in test/extended/
println("\n=== Phase 3: Feature Tests ===")

@safetestset "AD Backends (Smoke Test)" begin
    include("ad_backends_smoke.jl")
end
@safetestset "Subsampling" begin
    include("subsampling.jl")
end
@safetestset "Diagnostics" begin
    include("diagnostics.jl")
end
@safetestset "Alpha Extraction" begin
    include("alpha_extraction.jl")
end

# Phase 4: Integration tests (slowest, real data)
# These tests verify real data handling
println("\n=== Phase 4: Integration Tests ===")

@safetestset "Real Data (MCMC and VI)" begin
    include("real_data.jl")
end
