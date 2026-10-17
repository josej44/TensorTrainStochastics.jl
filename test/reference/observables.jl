# ============================================
# OBSERVABLES (accumulated on the fly)
# ============================================

"""
    compute_total_energy(spins, j_vector, h_vector)

E = -Σᵢ Jᵢ σᵢ σᵢ₊₁ - Σᵢ hᵢ σᵢ for ±1 spins.
"""
function compute_total_energy(spins, j_vector, h_vector)
    N = length(spins)
    energy = 0.0

    # Interaction term
    for i in 1:(N-1)
        energy -= j_vector[i] * spins[i] * spins[i+1]
    end

    # Field term
    for i in 1:N
        energy -= h_vector[i] * spins[i]
    end

    return energy
end

"""
    initialize_observable_accumulators(observables, N, T_steps, N_samples, n_chains)

Accumulators for `:magnetization`, `:energy`, `:nn_correlation`, `:overlap`, `:all_correlations`.
"""
function initialize_observable_accumulators(observables, N, T_steps, N_samples, n_chains)
    accumulators = Dict{Symbol, Any}()

    for obs in observables
        if obs == :magnetization
            # ⟨σᵢ(t)⟩ per chain: (n_chains, N, T_steps+1)
            accumulators[:magnetization] = zeros(Float64, n_chains, N, T_steps + 1)

        elseif obs == :energy
            # ⟨E(t)⟩ per chain: (n_chains, T_steps+1)
            accumulators[:energy] = zeros(Float64, n_chains, T_steps + 1)
            accumulators[:energy_sq] = zeros(Float64, n_chains, T_steps + 1)  # for the variance

        elseif obs == :nn_correlation
            # ⟨σᵢ σᵢ₊₁⟩ averaged over bonds: (n_chains, T_steps+1)
            accumulators[:nn_correlation] = zeros(Float64, n_chains, T_steps + 1)

        elseif obs == :overlap
            if n_chains >= 2
                # Q(t) = (1/N) Σᵢ ⟨σᵢ¹(t) σᵢ²(t)⟩ for every pair of chains
                n_pairs = div(n_chains * (n_chains - 1), 2)
                accumulators[:overlap] = zeros(Float64, n_pairs, T_steps + 1)
                accumulators[:overlap_pairs] = [(i, j) for i in 1:n_chains for j in (i+1):n_chains]
            end

        elseif obs == :all_correlations
            # ⟨σᵢ σⱼ⟩ for all pairs: (n_chains, N, N, T_steps+1) (memory-hungry)
            accumulators[:all_correlations] = zeros(Float64, n_chains, N, N, T_steps + 1)

        else
            @warn "Observable $obs not recognized, ignored"
        end
    end

    return accumulators
end

"""
    accumulate_observables!(accumulators, chains, energies, t, sample, observables, j_vector, h_vector, betas)

Add the contribution of the current state (time index `t`) to the accumulators,
using the precomputed `energies`.
"""
function accumulate_observables!(accumulators, chains, energies, t, sample, observables,
                                j_vector, h_vector, betas)
    N = length(chains[1])
    n_chains = length(chains)

    for obs in observables
        if obs == :magnetization
            for c in 1:n_chains
                for i in 1:N
                    accumulators[:magnetization][c, i, t] += chains[c][i]
                end
            end

        elseif obs == :energy
            for c in 1:n_chains
                E = energies[c]  # precomputed energy
                accumulators[:energy][c, t] += E
                accumulators[:energy_sq][c, t] += E^2
            end

        elseif obs == :nn_correlation
            for c in 1:n_chains
                nn_corr = 0.0
                for i in 1:(N-1)
                    nn_corr += chains[c][i] * chains[c][i+1]
                end
                accumulators[:nn_correlation][c, t] += nn_corr / (N - 1)
            end

        elseif obs == :overlap
            if n_chains >= 2
                pair_idx = 1
                for i in 1:n_chains
                    for j in (i+1):n_chains
                        overlap_ij = 0.0
                        for site in 1:N
                            overlap_ij += chains[i][site] * chains[j][site]
                        end
                        accumulators[:overlap][pair_idx, t] += overlap_ij / N
                        pair_idx += 1
                    end
                end
            end

        elseif obs == :all_correlations
            for c in 1:n_chains
                for i in 1:N
                    for j in 1:N
                        accumulators[:all_correlations][c, i, j, t] +=
                            chains[c][i] * chains[c][j]
                    end
                end
            end
        end
    end
end

"""
    finalize_observables(accumulators, N_samples, T_steps, n_chains)

Averages over samples; for the energy also the variance and the standard error.
With one chain the leading chain dimension is dropped.
"""
function finalize_observables(accumulators, N_samples, T_steps, n_chains)
    results = Dict{Symbol, Any}()

    for (obs_name, acc_data) in accumulators
        if obs_name == :magnetization
            mag = acc_data ./ N_samples
            results[:magnetization] = mag
            if n_chains == 1
                results[:magnetization] = mag[1, :, :]
            end

        elseif obs_name == :energy
            energy_mean = accumulators[:energy] ./ N_samples
            energy_sq_mean = accumulators[:energy_sq] ./ N_samples

            # Variance ⟨E²⟩ - ⟨E⟩² and standard error σ/√N_samples
            energy_var = energy_sq_mean .- energy_mean.^2
            energy_std = sqrt.(max.(energy_var, 0.0))
            energy_error = energy_std ./ sqrt(N_samples)

            results[:energy] = energy_mean
            results[:energy_error] = energy_error
            results[:energy_variance] = energy_var

            if n_chains == 1
                results[:energy] = energy_mean[1, :]
                results[:energy_error] = energy_error[1, :]
                results[:energy_variance] = energy_var[1, :]
            end

        elseif obs_name == :nn_correlation
            nn_corr = acc_data ./ N_samples
            results[:nn_correlation] = nn_corr
            if n_chains == 1
                results[:nn_correlation] = nn_corr[1, :]
            end

        elseif obs_name == :overlap
            overlap = acc_data ./ N_samples
            results[:overlap] = overlap
            if haskey(accumulators, :overlap_pairs)
                results[:overlap_pairs] = accumulators[:overlap_pairs]
            end

        elseif obs_name == :all_correlations
            corr = acc_data ./ N_samples
            results[:all_correlations] = corr
            if n_chains == 1
                results[:all_correlations] = corr[1, :, :, :]
            end
        end
    end

    return results
end
