# ============================================
# SWAP CRITERIA FOR PARALLEL TEMPERING
# ============================================

"""
    apply_fixed_rate_swap!(chains, s, rng) -> n_swaps

For every adjacent pair of chains (i, i+1), exchange the configurations with probability `s`.
"""
function apply_fixed_rate_swap!(chains, s, rng)
    n_chains = length(chains)
    n_swaps = 0

    for i in 1:(n_chains - 1)
        if rand(rng) < s
            chains[i], chains[i+1] = chains[i+1], chains[i]
            n_swaps += 1
        end
    end

    return n_swaps
end

# Exponent of the Metropolis swap test: the swap of chains i and j is accepted with
# probability min(1, exp(-delta_E_swap)).
# Bug A: the original used (βᵢ - βⱼ)(Eᵢ - Eⱼ), i.e. min(1, exp(-(βᵢ-βⱼ)(Eᵢ-Eⱼ))), which does not
# leave π_βᵢ ⊗ π_βⱼ invariant. The correct acceptance is min(1, exp((βᵢ-βⱼ)(Eᵢ-Eⱼ))).
_delta_E_swap(beta_i, beta_j, E_i, E_j) = (beta_j - beta_i) * (E_i - E_j)

"""
    apply_metropolis_swap!(chains, betas, j_vector, h_vector, rng) -> (n_attempts, n_accepted)

Metropolis swap of every adjacent pair of chains: accept with probability
min(1, exp((βᵢ - βᵢ₊₁)(Eᵢ - Eᵢ₊₁))) (energies recomputed).
"""
function apply_metropolis_swap!(chains, betas, j_vector, h_vector, rng)
    n_chains = length(chains)
    n_attempts = 0
    n_accepted = 0

    for i in 1:(n_chains - 1)
        n_attempts += 1

        E_i = compute_total_energy(chains[i], j_vector, h_vector)
        E_i_plus_1 = compute_total_energy(chains[i+1], j_vector, h_vector)

        delta_E_swap = _delta_E_swap(betas[i], betas[i+1], E_i, E_i_plus_1)

        if delta_E_swap <= 0
            chains[i], chains[i+1] = chains[i+1], chains[i]
            n_accepted += 1
        else
            if rand(rng) < exp(-delta_E_swap)
                chains[i], chains[i+1] = chains[i+1], chains[i]
                n_accepted += 1
            end
        end
    end

    return (n_attempts, n_accepted)
end

"""
    apply_metropolis_swap_with_energies!(chains, energies, betas, rng) -> (n_attempts, n_accepted)

Same as [`apply_metropolis_swap!`](@ref) with precomputed `energies` (swapped together
with the chains).
"""
function apply_metropolis_swap_with_energies!(chains, energies, betas, rng)
    n_chains = length(chains)
    n_attempts = 0
    n_accepted = 0

    for i in 1:(n_chains - 1)
        n_attempts += 1

        E_i = energies[i]
        E_i_plus_1 = energies[i+1]

        delta_E_swap = _delta_E_swap(betas[i], betas[i+1], E_i, E_i_plus_1)

        if delta_E_swap <= 0 || rand(rng) < exp(-delta_E_swap)
            chains[i], chains[i+1] = chains[i+1], chains[i]
            energies[i], energies[i+1] = energies[i+1], energies[i]
            n_accepted += 1
        end
    end

    return (n_attempts, n_accepted)
end

"""
    compute_swap_acceptance_rate(chains, betas, j_vector, h_vector, n_trials, rng) -> Vector{Float64}

Fraction of accepted Metropolis swaps for every adjacent pair over `n_trials` attempts
(on a copy of the chains).
"""
function compute_swap_acceptance_rate(chains, betas, j_vector, h_vector,
                                     n_trials::Int, rng)
    n_chains = length(chains)

    if n_chains < 2
        return Float64[]
    end

    chains_copy = [copy(chain) for chain in chains]

    acceptances = zeros(Int, n_chains - 1)

    for _ in 1:n_trials
        for i in 1:(n_chains - 1)
            E_i = compute_total_energy(chains_copy[i], j_vector, h_vector)
            E_i_plus_1 = compute_total_energy(chains_copy[i+1], j_vector, h_vector)

            delta_E_swap = _delta_E_swap(betas[i], betas[i+1], E_i, E_i_plus_1)

            if delta_E_swap <= 0 || rand(rng) < exp(-delta_E_swap)
                chains_copy[i], chains_copy[i+1] = chains_copy[i+1], chains_copy[i]
                acceptances[i] += 1
            end
        end
    end

    return acceptances ./ n_trials
end

"""
    suggest_beta_schedule(beta_min, beta_max, n_chains; target_acceptance=0.3, distribution=:geometric)

Inverse-temperature schedule for parallel tempering, geometric or linear between
`beta_min` and `beta_max`.
"""
function suggest_beta_schedule(beta_min::Float64, beta_max::Float64, n_chains::Int;
                              target_acceptance::Float64 = 0.3,
                              distribution::Symbol = :geometric)
    @assert beta_min < beta_max "beta_min must be smaller than beta_max"
    @assert n_chains >= 2 "at least 2 chains are needed"
    @assert 0 < target_acceptance < 1 "target_acceptance must be in (0,1)"

    if distribution == :geometric
        # Geometric: βᵢ₊₁/βᵢ constant
        ratio = (beta_max / beta_min)^(1/(n_chains - 1))
        return [beta_min * ratio^(i-1) for i in 1:n_chains]

    elseif distribution == :linear
        return range(beta_min, beta_max, length = n_chains) |> collect

    else
        error("distribution must be :geometric or :linear")
    end
end
