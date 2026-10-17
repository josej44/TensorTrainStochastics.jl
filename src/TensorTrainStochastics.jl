module TensorTrainStochastics

# ---- external dependencies (ONLY here) ----
using TensorTrains
using TensorTrains: compress!, TruncBond, TruncBondThresh
using TensorCast
using Tullio
using LogarithmicNumbers
using ProgressMeter
using LinearAlgebra
using Random
using Statistics

# ---- core ----
include("core/parameters.jl")
include("core/initialization.jl")
include("core/tt_utilities.jl")

# ---- lattices ----
include("lattices/graph_generation.jl")
include("lattices/graph_weights.jl")

# ---- distributions ----
include("distributions/boltzmann.jl")
include("distributions/initial_conditions.jl")

# ---- transitions ----
include("transitions/rates.jl")
include("transitions/chain_builder.jl")
include("transitions/graph_builder.jl")

# ---- evolution ----
include("evolution/evolve.jl")
include("evolution/swap.jl")

# ---- observables ----
include("observables/subproducts.jl")
include("observables/marginals.jl")
include("observables/correlations.jl")
include("observables/energy.jl")
include("observables/time_correlations.jl")
include("observables/clusters.jl")
include("observables/system_description.jl")

# ---- utils ----
include("utils.jl")

# =========================================================
#                       PUBLIC API
# =========================================================
# Each domain commit adds the names it defines.

# models
export AbstractSpinModel, ChainModel, GraphModel, nsites, local_fields, with_beta
export random_chain_model

# tensor-train utilities
export identity_tensor_train, divide_by_constant!, multiply_by_constant!, hadamard,
       replica_product, compose_transitions, k_step_transition_tt, multi_step_transition_tt,
       inverse_tt, inverse_secant_tt

# lattices
export random_3regular, reorder_rcm, add_gaussian_weights, build_instance

# distributions
export boltzmann_tt, boltzmann_swap_tt, boltzmann_swap_n_tt
export random_P0, parallel_random_P0_fixed, transform_init_cond_simple, transform_init_cond_multiple

# transitions
export glauber_transition_rate, metropolis_transition_rate, temp_zero_transition_rate,
       transition_rate_inertia
export build_parallel_transition_tt, build_sequential_transition_tt,
       build_transition_tensortrain, parallel_transition_tensor_train

# evolution
export distribution_b_tt, distribution_b_tt_exp
export ConstantSwap, EnergySwap, no_swap, tt_swap

end # module
