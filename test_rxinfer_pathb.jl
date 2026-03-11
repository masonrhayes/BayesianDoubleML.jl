# Quick test script for RxInfer Path B implementation
# Run this from the project root with: julia --project test_rxinfer_pathb.jl

using Pkg
Pkg.activate(".")

using BayesianDoubleML
using BayesianDoubleML.RxInferBDML  # Import the RxInfer submodule
using Random

println("="^60)
println("Testing RxInfer Path B Implementation")
println("="^60)

# Generate test data
n, p = 100, 5
alpha_true = 0.5
println("\nGenerating data: n=$n, p=$p, true α=$alpha_true")
Y, D, X, alpha_true_actual, params = generate_dgp_table1(n, p, 2.0; alpha_true=alpha_true, rng=MersenneTwister(999))
println("Data generated successfully")
println("  Y range: [$(round(minimum(Y), digits=2)), $(round(maximum(Y), digits=2))]")
println("  D range: [$(round(minimum(D), digits=2)), $(round(maximum(D), digits=2))]")
println("  X size: $(size(X))")

# Test the RxInfer implementation
println("\n" * "="^60)
println("Running rx_infer() with 50 iterations...")
println("="^60)

try
    result = rx_infer(Y, D, X; iterations=50)
    
    println("\n✓ Inference completed!")
    println("\nResults:")
    println(result)
    
    println("\nComparison with true value:")
    println("  True α:        $alpha_true")
    println("  Estimated α:   $(round(result.alpha_mean, digits=4))")
    println("  Error:         $(round(abs(result.alpha_mean - alpha_true), digits=4))")
    println("  Within 95% CI: $(alpha_true >= credible_interval(result, 0.95)[1] && alpha_true <= credible_interval(result, 0.95)[2])")
    
    if result.converged
        println("\n✓ VMP converged successfully")
    else
        println("\n⚠ VMP did not converge (may need more iterations)")
    end
    
catch e
    println("\n✗ Error during inference:")
    println(e)
    
    # Print stacktrace for debugging
    println("\nStacktrace:")
    Base.show_backtrace(stdout, catch_backtrace())
    println()
end

println("\n" * "="^60)
println("Test complete")
println("="^60)
