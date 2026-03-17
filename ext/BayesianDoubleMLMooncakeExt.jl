module BayesianDoubleMLMooncakeExt

using BayesianDoubleML
using Mooncake
using ADTypes

# This extension is loaded when Mooncake is available in the environment.
# It ensures that Mooncake AD backend can be used with BayesianDoubleML.

# Mooncake-specific initialization or optimizations can be added here.
# Currently, DifferentiationInterface handles the AD backend selection,
# so this extension primarily ensures Mooncake is loaded when AutoMooncake is used.

function __init__()
    return @debug "BayesianDoubleMLMooncakeExt loaded - Mooncake AD backend is available"
end

end
