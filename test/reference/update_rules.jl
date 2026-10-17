# ============================================
# UPDATE RULES
# ============================================
# Time units: :sequential / :metropolis / :zero_temp update ONE random site per step;
# :parallel updates all sites (T_sequential ≈ N × T_parallel).

"""
    glauber_transition_rate(sigma_neighbors, sigma_new, site_index, beta, j_vector, h_vector, p0)

Glauber probability P(σᵢᵗ⁺¹ = sigma_new | neighbourhood), with inertia `p0`.

Layout of `sigma_neighbors`: site 1 `[σ₁, σ₂]`, site N `[σₙ₋₁, σₙ]`, middle `[σᵢ₋₁, σᵢ, σᵢ₊₁]`.
"""
function glauber_transition_rate(sigma_neighbors, sigma_new, site_index, beta,
                                j_vector, h_vector, p0)
    N = length(h_vector)

    if site_index == 1
        # Site 1: right neighbour only
        h_eff = j_vector[1] * sigma_neighbors[2] + h_vector[1]
        sigma_current = sigma_neighbors[1]

    elseif site_index == N
        # Site N: left neighbour only
        h_eff = j_vector[end] * sigma_neighbors[1] + h_vector[end]
        sigma_current = sigma_neighbors[2]

    else
        # Middle site
        h_eff = j_vector[site_index - 1] * sigma_neighbors[1] +
                j_vector[site_index] * sigma_neighbors[3] +
                h_vector[site_index]
        sigma_current = sigma_neighbors[2]
    end

    # Glauber dynamics: P(σ → σ') ∝ exp(β σ' h_eff)
    glauber_prob = exp(beta * sigma_new * h_eff) / (2 * cosh(beta * h_eff))

    # Keep-the-state component (if p0 > 0)
    if sigma_new == sigma_current
        return (1 - p0) * glauber_prob + p0
    else
        return (1 - p0) * glauber_prob
    end
end

"""
    compute_local_energy_change(spins, site_index, j_vector, h_vector)

ΔE = E(flip) - E(current) = 2 σᵢ h_eff for flipping the spin at `site_index`,
with h_eff = Jᵢ₋₁ σᵢ₋₁ + Jᵢ σᵢ₊₁ + hᵢ.
"""
function compute_local_energy_change(spins, site_index, j_vector, h_vector)
    N = length(spins)
    sigma_i = spins[site_index]

    # Effective field
    h_eff = h_vector[site_index]

    if site_index > 1
        h_eff += j_vector[site_index - 1] * spins[site_index - 1]
    end

    if site_index < N
        h_eff += j_vector[site_index] * spins[site_index + 1]
    end

    return 2 * sigma_i * h_eff
end

# ============================================
# PARALLEL UPDATE (GLAUBER)
# ============================================

"""
    parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)

Parallel Glauber update: all spins are drawn simultaneously from the configuration at
time t; the result is written to `spins_new`.
"""
function parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)
    N = length(spins)

    for i in 1:N
        if i == 1
            sigma_neighbors = [spins[1], spins[2]]
        elseif i == N
            sigma_neighbors = [spins[N-1], spins[N]]
        else
            sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
        end

        p_up = glauber_transition_rate(sigma_neighbors, 1, i, beta, j_vector, h_vector, p0)
        p_down = glauber_transition_rate(sigma_neighbors, -1, i, beta, j_vector, h_vector, p0)

        # Normalize (numerical safety)
        p_total = p_up + p_down
        p_up /= p_total

        spins_new[i] = rand(rng) < p_up ? 1 : -1
    end
end

# ============================================
# SEQUENTIAL UPDATE (GLAUBER)
# ============================================

"""
    sequential_update!(spins, beta, j_vector, h_vector, p0, rng)

Sequential Glauber update of ONE random site (p0 is forced to 0, as in the original).
"""
function sequential_update!(spins, beta, j_vector, h_vector, p0, rng)
    N = length(spins)

    i = rand(rng, 1:N)

    if i == 1
        sigma_neighbors = [spins[1], spins[2]]
    elseif i == N
        sigma_neighbors = [spins[N-1], spins[N]]
    else
        sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
    end

    # Note: p0 = 0 is forced in sequential Glauber
    p_up = glauber_transition_rate(sigma_neighbors, 1, i, beta, j_vector, h_vector, 0.0)
    p_down = glauber_transition_rate(sigma_neighbors, -1, i, beta, j_vector, h_vector, 0.0)

    p_total = p_up + p_down
    p_up /= p_total

    spins[i] = rand(rng) < p_up ? 1 : -1
end

# ============================================
# METROPOLIS UPDATE
# ============================================

"""
    metropolis_update!(spins, beta, j_vector, h_vector, rng) -> accepted::Bool

Metropolis update of ONE random site: flip with probability min(1, exp(-β ΔE)).
"""
function metropolis_update!(spins, beta, j_vector, h_vector, rng)
    N = length(spins)

    i = rand(rng, 1:N)

    delta_E = compute_local_energy_change(spins, i, j_vector, h_vector)

    accepted = false
    if delta_E <= 0
        spins[i] = -spins[i]
        accepted = true
    else
        if rand(rng) < exp(-beta * delta_E)
            spins[i] = -spins[i]
            accepted = true
        end
    end

    return accepted
end

"""
    zero_update!(spins, beta, j_vector, h_vector, rng) -> accepted::Bool

Zero-temperature update of ONE random site: flip iff ΔE ≤ 0.
"""
function zero_update!(spins, beta, j_vector, h_vector, rng)
    N = length(spins)

    i = rand(rng, 1:N)

    delta_E = compute_local_energy_change(spins, i, j_vector, h_vector)

    accepted = false
    if delta_E <= 0
        spins[i] = -spins[i]
        accepted = true
    end

    return accepted
end

"""
    apply_update!(spins, spins_new, update_rule, beta, j_vector, h_vector, p0, rng)

Apply `update_rule` (`:parallel`, `:sequential`, `:metropolis`, `:zero_temp`).
`spins_new` is only used by `:parallel`. Returns `accepted` for Metropolis/zero-temperature.
"""
function apply_update!(spins, spins_new, update_rule::Symbol, beta,
                      j_vector, h_vector, p0, rng)
    if update_rule == :parallel
        parallel_update!(spins_new, spins, beta, j_vector, h_vector, p0, rng)
        return nothing
    elseif update_rule == :sequential
        sequential_update!(spins, beta, j_vector, h_vector, p0, rng)
        return nothing
    elseif update_rule == :metropolis
        accepted = metropolis_update!(spins, beta, j_vector, h_vector, rng)
        return accepted
    elseif update_rule == :zero_temp
        accepted = zero_update!(spins, beta, j_vector, h_vector, rng)
        return accepted
    else
        error("unknown update_rule: $update_rule. Options: :parallel, :sequential, :metropolis, :zero_temp")
    end
end

# ============================================
# UPDATES THAT ALSO RETURN THE ENERGY CHANGE
# ============================================

"Sequential Glauber update of one random site; returns the energy change."
function sequential_update_with_energy!(spins, beta, j_vector, h_vector, p0, rng)
    N = length(spins)
    i = rand(rng, 1:N)

    old_spin = spins[i]

    # Effective field at site i
    h_eff = h_vector[i]
    if i > 1
        h_eff += j_vector[i - 1] * spins[i - 1]
    end
    if i < N
        h_eff += j_vector[i] * spins[i + 1]
    end

    if i == 1
        sigma_neighbors = [spins[1], spins[2]]
    elseif i == N
        sigma_neighbors = [spins[N-1], spins[N]]
    else
        sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
    end

    p_up = glauber_transition_rate(sigma_neighbors, 1, i, beta, j_vector, h_vector, 0.0)
    p_down = glauber_transition_rate(sigma_neighbors, -1, i, beta, j_vector, h_vector, 0.0)

    p_total = p_up + p_down
    p_up /= p_total

    new_spin = rand(rng) < p_up ? 1 : -1
    spins[i] = new_spin

    # ΔE = E_new - E_old = -(new_spin - old_spin) * h_eff
    delta_E = -(new_spin - old_spin) * h_eff

    return delta_E
end

"Metropolis update of one random site; returns `(accepted, ΔE)`."
function metropolis_update_with_energy!(spins, beta, j_vector, h_vector, rng)
    N = length(spins)
    i = rand(rng, 1:N)

    delta_E = compute_local_energy_change(spins, i, j_vector, h_vector)

    accepted = false
    if delta_E <= 0
        spins[i] = -spins[i]
        accepted = true
    else
        if rand(rng) < exp(-beta * delta_E)
            spins[i] = -spins[i]
            accepted = true
        end
    end

    return accepted, (accepted ? delta_E : 0.0)
end

"Zero-temperature update of one random site; returns `(accepted, ΔE)`."
function zero_update_with_energy!(spins, beta, j_vector, h_vector, rng)
    N = length(spins)
    i = rand(rng, 1:N)

    delta_E = compute_local_energy_change(spins, i, j_vector, h_vector)

    accepted = false
    if delta_E <= 0
        spins[i] = -spins[i]
        accepted = true
    end

    return accepted, (accepted ? delta_E : 0.0)
end
