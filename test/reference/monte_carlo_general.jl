# ============================================
# GENERAL MONTE CARLO DRIVER
# ============================================

"""
    run_monte_carlo_general(params::MCParameters; initial_probs, T_steps, update_rule=:parallel,
                            swap_criterion=nothing, observables=[:magnetization, :energy],
                            save_trajectory=false, N_samples=1000, debug=false)

Monte Carlo simulation of `n_chains(params)` Ising chains, all started from the same
random configuration (`initial_probs[i] = P(σᵢ = +1)`), averaged over `N_samples`
independent runs of `T_steps` steps.

- `update_rule`: `:parallel`, `:sequential`, `:metropolis` or `:zero_temp`.
- `swap_criterion`: `nothing`, `:fixed_rate` (probability `params.s`) or `:metropolis`;
  the swap is applied after every update step.
- `observables`: any of `:magnetization`, `:energy`, `:nn_correlation`, `:overlap`,
  `:all_correlations`.

Returns a `Dict` with `:observables` (see [`finalize_observables`](@ref)), `:params`
and, if requested, `:trajectories`.
"""
function run_monte_carlo_general(
    params::MCParameters;
    initial_probs::Vector{Float64},
    T_steps::Int,
    update_rule::Symbol = :parallel,
    swap_criterion::Union{Nothing, Symbol} = nothing,
    observables::Vector{Symbol} = [:magnetization, :energy],
    save_trajectory::Bool = false,
    N_samples::Int = 1000,
    debug::Bool = false
)
    # ============================================
    # VALIDATION
    # ============================================

    validate(params)

    N = params.N
    n_chains_val = n_chains(params)

    @assert length(initial_probs) == N "initial_probs must have length N"
    @assert update_rule in [:parallel, :sequential, :metropolis, :zero_temp] "update_rule must be :parallel, :sequential, :metropolis or :zero_temp"

    if swap_criterion !== nothing
        @assert n_chains_val >= 2 "swap_criterion needs at least 2 chains"
        @assert swap_criterion in [:fixed_rate, :metropolis] "swap_criterion must be :fixed_rate or :metropolis"
    end

    # ============================================
    # INITIALIZATION
    # ============================================

    rng = MersenneTwister(params.seed)

    # All chains start from the same configuration
    initialization = initialize_spins(N, initial_probs, rng)
    chains = [copy(initialization) for _ in 1:n_chains_val]

    chain_energies = [compute_total_energy(chains[c], params.j_vector, params.h_vector)
                      for c in 1:n_chains_val]

    # Buffers only needed for the parallel update
    if update_rule == :parallel
        chains_new = [zeros(Int, N) for _ in 1:n_chains_val]
    else
        chains_new = nothing
    end

    obs_accumulators = initialize_observable_accumulators(observables, N, T_steps,
                                                          N_samples, n_chains_val)

    trajectories = save_trajectory ? [zeros(Int, N_samples, N, T_steps + 1) for _ in 1:n_chains_val] : nothing

    # ============================================
    # SAMPLING LOOP
    # ============================================

    for sample in 1:N_samples
        # Re-initialize all chains
        initialization = initialize_spins(N, initial_probs, rng)
        for c in 1:n_chains_val
            chains[c] .= initialization
            chain_energies[c] = compute_total_energy(chains[c], params.j_vector, params.h_vector)
        end

        if save_trajectory
            for c in 1:n_chains_val
                trajectories[c][sample, :, 1] .= chains[c]
            end
        end

        accumulate_observables!(obs_accumulators, chains, chain_energies, 1, sample, observables,
                               params.j_vector, params.h_vector, params.betas)

        # ============================================
        # TIME EVOLUTION
        # ============================================

        for t in 1:T_steps
            if update_rule == :parallel
                for c in 1:n_chains_val
                    parallel_update!(chains_new[c], chains[c], params.betas[c],
                                    params.j_vector, params.h_vector, params.p0, rng)
                    chains[c] .= chains_new[c]

                    chain_energies[c] = compute_total_energy(chains[c], params.j_vector, params.h_vector)
                end

            elseif update_rule == :sequential
                for c in 1:n_chains_val
                    delta_E = sequential_update_with_energy!(chains[c], params.betas[c],
                                                 params.j_vector, params.h_vector, params.p0, rng)
                    chain_energies[c] += delta_E
                end

            elseif update_rule == :zero_temp
                for c in 1:n_chains_val
                    accepted, delta_E = zero_update_with_energy!(chains[c], params.betas[c],
                                                           params.j_vector, params.h_vector, rng)
                    if accepted
                        chain_energies[c] += delta_E
                    end
                end

            else  # :metropolis
                for c in 1:n_chains_val
                    accepted, delta_E = metropolis_update_with_energy!(chains[c], params.betas[c],
                                                           params.j_vector, params.h_vector, rng)
                    if accepted
                        chain_energies[c] += delta_E
                    end
                end
            end

            # Swap between chains (if any)
            if swap_criterion !== nothing && n_chains_val >= 2
                if swap_criterion == :fixed_rate
                    n_swaps = apply_fixed_rate_swap!(chains, params.s, rng)

                    if n_swaps > 0
                        for c in 1:n_chains_val
                            chain_energies[c] = compute_total_energy(chains[c], params.j_vector, params.h_vector)
                        end
                    end

                else  # :metropolis
                    apply_metropolis_swap_with_energies!(chains, chain_energies, params.betas, rng)
                end
            end

            if save_trajectory
                for c in 1:n_chains_val
                    trajectories[c][sample, :, t + 1] .= chains[c]
                end
            end

            if debug && sample == 1 && t <= 3
                println("Sample $sample, t=$t")
                for c in 1:min(2, n_chains_val)
                    println("  Chain $c: ", chains[c][1:min(5, N)])
                    println("  Energy $c: ", chain_energies[c])
                end
            end

            accumulate_observables!(obs_accumulators, chains, chain_energies, t + 1, sample, observables,
                       params.j_vector, params.h_vector, params.betas)
        end
    end

    # ============================================
    # RESULTS
    # ============================================

    computed_observables = finalize_observables(obs_accumulators, N_samples, T_steps, n_chains_val)

    result = Dict{Symbol, Any}(
        :observables => computed_observables,
        :params => params
    )

    if save_trajectory
        result[:trajectories] = trajectories
    end

    return result
end

"""
    initialize_spins(N, initial_probs, rng)

Random ±1 configuration with P(σᵢ = +1) = initial_probs[i].
"""
function initialize_spins(N::Int, initial_probs::Vector{Float64}, rng)
    spins = zeros(Int, N)
    for i in 1:N
        spins[i] = rand(rng) < initial_probs[i] ? 1 : -1
    end
    return spins
end
