module BayesianDoubleMLZygoteExt

using BayesianDoubleML
using Zygote
using ADTypes

# This extension is loaded when Zygote is available in the environment.
# It ensures that Zygote AD backend can be used with BayesianDoubleML.

# Zygote-specific initialization or optimizations can be added here.
# Currently, DifferentiationInterface handles the AD backend selection,
# so this extension primarily ensures Zygote is loaded when AutoZygote is used.

function __init__()
    @debug "BayesianDoubleMLZygoteExt loaded - Zygote AD backend is available"
end

end