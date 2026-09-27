module TensorTrainStochastics

# ---- dependencias externas (SOLO aquí) ----
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
#                       API PÚBLICA
# =========================================================

# parámetros
export MCParameters, random_params, parallel_random_params

# inicialización / instancias
export random_P0, transform_init_cond_simple, transform_init_cond_multiple
export random_3regular, reorder_rcm, add_gaussian_weights, build_instance

# distribuciones
export boltzmann_tt, boltzmann_swap_tt, boltzmann_swap_n_tt

# transiciones
export build_transition_tt, build_transition_tensortrain,
       build_sequential_transition_tt, build_parallel_transition_tt,
       parallel_transition_tensor_train,
       k_step_transition_tt, multi_step_transition_tt

export glauber_transition_rate, metropolis_transition_rate,
       temp_zero_transition_rate, transition_rate_inertia

# evolución
export distribution_b_tt, distribution_b_tt_exp
export ConstantSwap, EnergySwap, no_swap, tt_swap

# observables
export full_marginals_system, full_second_order_marginals_system,
       marginal_ev_system, second_moment_system,
       correlation_between_spins_system, correlation_next_pairs_simple,
       full_simple_ev_system
export energy_function, energy_function_simple
export full_tcorr_system, times_correlation_evolution
export cluster_size_prob, portion_of_TT
export system_description, system_description_over_time,
       system_description_over_time_simple

end # module
