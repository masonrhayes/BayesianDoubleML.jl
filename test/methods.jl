# Inference Methods Tests
# Tests for MCMCMethod, UnifiedVIMethod, SimpleVIMethod constructors and traits

using BayesianDoubleML
using Test

@testset "MCMCMethod Constructors" begin
    # Default NUTS
    method = MCMCMethod(:nuts)
    @test method.algorithm == :nuts
    @test method.target_acceptance ≈ 0.8
    @test method.max_depth == 10

    # Custom NUTS settings
    method_custom = MCMCMethod(:nuts; target_acceptance = 0.9, max_depth = 12)
    @test method_custom.target_acceptance ≈ 0.9
    @test method_custom.max_depth == 12

    # HMC
    method_hmc = MCMCMethod(:hmc; leapfrog_steps = 20, step_size = 0.05)
    @test method_hmc.algorithm == :hmc
    @test method_hmc.leapfrog_steps == 20
    @test method_hmc.step_size ≈ 0.05

    # Convenience constructors
    nuts = MCMCNUTS()
    @test nuts.algorithm == :nuts
    @test nuts.target_acceptance ≈ 0.8

    nuts_custom = MCMCNUTS(; target_acceptance = 0.85, max_depth = 11)
    @test nuts_custom.target_acceptance ≈ 0.85
    @test nuts_custom.max_depth == 11

    hmc = MCMCHMC()
    @test hmc.algorithm == :hmc
    @test hmc.leapfrog_steps == 10
    @test hmc.step_size ≈ 0.1
end

@testset "MCMCMethod Validation" begin
    # Invalid target acceptance
    @test_throws AssertionError MCMCMethod(:nuts; target_acceptance = 1.5)
    @test_throws AssertionError MCMCMethod(:nuts; target_acceptance = -0.1)
    @test_throws AssertionError MCMCMethod(:nuts; target_acceptance = 0.0)

    # Invalid max_depth
    @test_throws AssertionError MCMCMethod(:nuts; max_depth = 0)
    @test_throws AssertionError MCMCMethod(:nuts; max_depth = -5)

    # Invalid leapfrog_steps
    @test_throws AssertionError MCMCMethod(:hmc; leapfrog_steps = 0)
    @test_throws AssertionError MCMCMethod(:hmc; leapfrog_steps = -10)

    # Invalid step_size
    @test_throws AssertionError MCMCMethod(:hmc; step_size = -0.1)
    @test_throws AssertionError MCMCMethod(:hmc; step_size = 0.0)

    # Invalid algorithm
    @test_throws ArgumentError MCMCMethod(:invalid)
    @test_throws ArgumentError MCMCMethod(:nuts_custom)
end

@testset "UnifiedVIMethod Constructors" begin
    # Default
    method = UnifiedVIMethod()
    @test method.ad_backend == AutoReverseDiff
    @test method.subsample === nothing
    @test method.batch_size == -1
    @test method.n_montecarlo == 10

    # Custom settings
    method_custom = UnifiedVIMethod(;
        ad_backend = AutoMooncake,
        subsample = true,
        batch_size = 128,
        n_montecarlo = 20
    )
    @test method_custom.ad_backend == AutoMooncake
    @test method_custom.subsample == true
    @test method_custom.batch_size == 128
    @test method_custom.n_montecarlo == 20

    # Convenience constructor
    unified = UnifiedVI()
    @test unified isa UnifiedVIMethod
    @test unified.ad_backend == AutoReverseDiff
end

@testset "UnifiedVIMethod Validation" begin
    # Invalid n_montecarlo
    @test_throws AssertionError UnifiedVIMethod(; n_montecarlo = 0)
    @test_throws AssertionError UnifiedVIMethod(; n_montecarlo = -5)
end

@testset "SimpleVIMethod Constructors" begin
    # Default (Mooncake)
    method = SimpleVIMethod()
    @test method.ad_backend == AutoMooncake

    # Custom AD backend
    method_rd = SimpleVIMethod(; ad_backend = AutoReverseDiff)
    @test method_rd.ad_backend == AutoReverseDiff

    # Convenience constructor
    simple = SimpleVI()
    @test simple isa SimpleVIMethod
    @test simple.ad_backend == AutoMooncake
end

@testset "Method Traits - uses_sampling" begin
    # All methods use sampling
    @test BayesianDoubleML.uses_sampling(MCMCMethod(:nuts)) == true
    @test BayesianDoubleML.uses_sampling(MCMCMethod(:hmc)) == true
    @test BayesianDoubleML.uses_sampling(UnifiedVIMethod()) == true
    @test BayesianDoubleML.uses_sampling(SimpleVIMethod()) == true
end

@testset "Method Traits - supports_subsampling" begin
    # Only UnifiedVIMethod supports subsampling
    @test BayesianDoubleML.supports_subsampling(MCMCMethod(:nuts)) == false
    @test BayesianDoubleML.supports_subsampling(MCMCMethod(:hmc)) == false
    @test BayesianDoubleML.supports_subsampling(UnifiedVIMethod()) == true
    @test BayesianDoubleML.supports_subsampling(SimpleVIMethod()) == false
end

@testset "Method Traits - is_deterministic" begin
    # No methods are deterministic (all use random sampling)
    @test BayesianDoubleML.is_deterministic(MCMCMethod(:nuts)) == false
    @test BayesianDoubleML.is_deterministic(UnifiedVIMethod()) == false
    @test BayesianDoubleML.is_deterministic(SimpleVIMethod()) == false
end

@testset "Method Traits - default_n_samples" begin
    # All methods default to 2000 samples
    @test BayesianDoubleML.default_n_samples(MCMCMethod(:nuts)) == 2000
    @test BayesianDoubleML.default_n_samples(UnifiedVIMethod()) == 2000
    @test BayesianDoubleML.default_n_samples(SimpleVIMethod()) == 2000
end

@testset "Method Traits - default_n_iterations" begin
    # All methods default to 1000 iterations
    @test BayesianDoubleML.default_n_iterations(MCMCMethod(:nuts)) == 1000
    @test BayesianDoubleML.default_n_iterations(UnifiedVIMethod()) == 1000
    @test BayesianDoubleML.default_n_iterations(SimpleVIMethod()) == 1000
end

@testset "Method Type Stability" begin
    # Convenience constructors should return MCMCMethod objects
    method_nuts = MCMCNUTS()
    @test typeof(method_nuts) == MCMCMethod
    @test method_nuts.algorithm == :nuts

    method_hmc = MCMCHMC()
    @test typeof(method_hmc) == MCMCMethod
    @test method_hmc.algorithm == :hmc

    method_unified = UnifiedVI()
    @test typeof(method_unified) == UnifiedVIMethod

    method_simple = SimpleVI()
    @test typeof(method_simple) == SimpleVIMethod
end
