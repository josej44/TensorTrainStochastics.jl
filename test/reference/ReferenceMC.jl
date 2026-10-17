"""
    ReferenceMC

Monte Carlo reference implementation (NOT public API), migrated from
`GlauberPT.jl/src/monte_carlo_efficient/` and the notebooks. Used only by the tests to
validate the tensor-train results. Spins are ±1 integers unless stated otherwise.
"""
module ReferenceMC

using Random, Statistics
using TensorTrainStochastics: ChainModel, nsites

include("parameters.jl")
include("update_rules.jl")
include("swap_criteria.jl")
include("observables.jl")
include("monte_carlo_general.jl")
include("notebook_tools.jl")

end # module
