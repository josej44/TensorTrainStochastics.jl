# Energies: expectation values from a distribution TT, and the exact ground state.

"""
    energy_function(full_expected_values, full_second_moments, model::ChainModel) -> (E₁, E₂)

Expected energies of the two replicas from [`marginal_ev_system`](@ref) and
[`second_moment_system`](@ref) (both replicas share the couplings of `model`).

# Example
```julia
E1, E2 = energy_function(ev, sm, m1)
```
"""
function energy_function(full_expected_values, full_second_moments, model::ChainModel)
    energy_1 = 0.0
    energy_2 = 0.0
    h_vector = model.h
    j_vector = model.J
    N = nsites(model)
    for k in 1:N
        energy_1 += -h_vector[k] * full_expected_values[k][1]
        energy_2 += -h_vector[k] * full_expected_values[k][2]
        if k < N
            energy_1 += -j_vector[k] * full_second_moments[k][1]
            energy_2 += -j_vector[k] * full_second_moments[k][2]
        end
    end
    return (energy_1, energy_2)
end

"""
    energy_function_simple(bt, model::ChainModel) -> Float64

Expected energy `⟨E⟩` of a single-chain normalized distribution. Requires `N ≥ 3`.

# Example
```julia
E = energy_function_simple(B, m)
```
"""
function energy_function_simple(bt, model::ChainModel)
    bb = full_subproducts_system(bt)
    second_marg = full_second_order_marginals_system(bt, bb, true)
    first_marg = full_marginals_system(bt, bb, true)

    full_ev_system = [p[2] - p[1] for p in first_marg]
    correlation_next_pairs = [p[1] + p[4] - p[2] - p[3] for p in second_marg]
    energy_1 = 0.0
    h_vector = model.h
    j_vector = model.J
    for k in 1:length(h_vector)
        energy_1 += -h_vector[k] * full_ev_system[k][1]
        if k < length(h_vector)
            energy_1 += -j_vector[k] * correlation_next_pairs[k][1]
        end
    end
    return energy_1
end

"""
    min_energy_config(model::ChainModel) -> (config, spins_pm, energy)

Exact ground state of the chain by dynamic programming (Viterbi):
`config` with indices (1 → -1, 2 → +1), `spins_pm` with values ±1.0, and the minimum energy.

# Example
```julia
config, spins, E0 = min_energy_config(ChainModel(ones(4), zeros(5), 1.0))
```
"""
function min_energy_config(model::ChainModel)
    J = model.J
    H = model.h
    n = length(H)

    # States: 1 -> spin -1,  2 -> spin +1
    spin = [-1.0, 1.0]

    # V[i, s] = minimum energy of sites 1..i with spin s at site i
    V    = fill(Inf, n, 2)
    back = fill(0,   n, 2)   # for the traceback

    # Initialization
    for s in 1:2
        V[1, s] = -H[1] * spin[s]
    end

    # Forward pass
    for i in 2:n
        for s in 1:2
            for sp in 1:2   # previous state
                cost = V[i-1, sp] - J[i-1] * spin[sp] * spin[s] - H[i] * spin[s]
                if cost < V[i, s]
                    V[i, s]    = cost
                    back[i, s] = sp
                end
            end
        end
    end

    # Traceback
    config = Vector{Int}(undef, n)
    config[n] = argmin(V[n, :])
    for i in n:-1:2
        config[i-1] = back[i, config[i]]
    end

    energy   = V[n, config[n]]
    spins_pm = [spin[s] for s in config]   # vector of -1.0 / +1.0

    return config, spins_pm, energy
end
