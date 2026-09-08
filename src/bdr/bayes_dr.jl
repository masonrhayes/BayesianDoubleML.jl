# Bayes-DR is split into data/result types, the two nuisance-model Gibbs
# samplers, and assembly of the posterior-corrected ATE inference procedure.
include("bayes_dr_types.jl")
include("bayes_dr_sampler.jl")
include("bayes_dr_fit.jl")
