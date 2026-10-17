# Single-site transition rates of an open chain.
#
# Common signature: rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)
# returns P(σᵢᵗ⁺¹ = sigma_new | neighbourhood at time t), where
#   site 1:            sigma_neighbors = [σ₁, σ₂]
#   site i (middle):   sigma_neighbors = [σᵢ₋₁, σᵢ, σᵢ₊₁]
#   site N:            sigma_neighbors = [σₙ₋₁, σₙ]
# and all spin values are physical (±1).

"""
    transition_rate_inertia(sigma_neighbors, sigma_new, site_index, model::ChainModel; p0=0.0)

Glauber rate with inertia `p0`: with probability `p0` the spin keeps its value,
otherwise it is drawn from the Glauber rule,

    P = (1 - p0) · exp(β σ' h_eff) / (2 cosh(β h_eff)) + p0 · [σ' == σᵢ] .

# Arguments
- `sigma_neighbors`: neighbourhood of the site (see the file header for the layout).
- `sigma_new`: new spin value (±1).
- `site_index`: site `i ∈ 1:N`.
- `model::ChainModel`: couplings, fields and `β`.

# Keywords
- `p0::Real`: inertia, in `[0, 1]`.

# Example
```julia
m = ChainModel(ones(2), zeros(3), 1.0)
transition_rate_inertia([1, 1, -1], 1, 2, m; p0 = 0.1)
```
"""
function transition_rate_inertia(sigma_neighbors, sigma_new, site_index, model::ChainModel; p0::Real = 0.0)
    N = nsites(model)
    betas = model.β
    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = model.J[1] * sigma_neighbors[2] + model.h[1]

    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = model.J[end] * sigma_neighbors[1] + model.h[end]

    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = model.J[site_index - 1] * sigma_neighbors[1] +
                model.J[site_index] * sigma_neighbors[3] +
                model.h[site_index]
    end

    # Glauber dynamics with inertia
    return (exp(betas * sigma_new * h_eff) / (2 * cosh(betas * h_eff)))*(1-p0) + p0* (sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2] ? 1.0 : 0.0)
end

"""
    glauber_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)

Glauber (heat-bath) rate `exp(β σ' h_eff) / (2 cosh(β h_eff))`, with
`h_eff = J_{i-1} σᵢ₋₁ + J_i σᵢ₊₁ + h_i`.

# Example
```julia
glauber_transition_rate([1, -1], -1, 1, ChainModel([0.5], [0.0, 0.0], 2.0))
```
"""
function glauber_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)
    N = nsites(model)

    betas = model.β

    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = model.J[1] * sigma_neighbors[2] + model.h[1]

    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = model.J[end] * sigma_neighbors[1] + model.h[end]

    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = model.J[site_index - 1] * sigma_neighbors[1] +
                model.J[site_index] * sigma_neighbors[3] +
                model.h[site_index]
    end

    # Glauber dynamics
    return (exp(betas * sigma_new * h_eff) / (2 * cosh(betas * h_eff)))
end

"""
    metropolis_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)

Metropolis rate: the spin flips with probability `min(1, exp(-β ΔE))`, where
`ΔE = 2 σᵢ h_eff` is the energy change of the flip.

# Example
```julia
metropolis_transition_rate([1, 1, 1], -1, 2, ChainModel(ones(2), zeros(3), 1.0))   # exp(-4)
```
"""
function metropolis_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)
    N = nsites(model)
    betas = model.β

    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = model.J[1] * sigma_neighbors[2] + model.h[1]
        delta_E = 2 * sigma_neighbors[1] * h_eff  # explicit factor 2

    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = model.J[end] * sigma_neighbors[1] + model.h[end]
        delta_E = 2 * sigma_neighbors[2] * h_eff  # explicit factor 2
    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = model.J[site_index - 1] * sigma_neighbors[1] +
                model.J[site_index] * sigma_neighbors[3] +
                model.h[site_index]
        delta_E = 2 * sigma_neighbors[2] * h_eff  # explicit factor 2
    end

    if sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2]
        return 1 - min(exp(-betas * delta_E), 1.0)
    else
        return min(exp(-betas * delta_E), 1.0)
    end
end

# 1 if the flip is accepted at zero temperature (ΔE ≤ 0), else 0.
function _zero_dec(x)
    if x > 0
        return 0
    else
        return 1
    end
end

"""
    temp_zero_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)

Zero-temperature rate: the spin flips with probability 1 if the flip does not raise
the energy (`ΔE ≤ 0`, ties included) and 0 otherwise. `model.β` is ignored.

# Example
```julia
temp_zero_transition_rate([1, -1, -1], 1, 2, ChainModel(ones(2), zeros(3), 1.0))   # 0
```
"""
function temp_zero_transition_rate(sigma_neighbors, sigma_new, site_index, model::ChainModel)
    N = nsites(model)

    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = model.J[1] * sigma_neighbors[2] + model.h[1]
        delta_E = 2 * sigma_neighbors[1] * h_eff  # explicit factor 2

    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = model.J[end] * sigma_neighbors[1] + model.h[end]
        delta_E = 2 * sigma_neighbors[2] * h_eff  # explicit factor 2
    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = model.J[site_index - 1] * sigma_neighbors[1] +
                model.J[site_index] * sigma_neighbors[3] +
                model.h[site_index]
        delta_E = 2 * sigma_neighbors[2] * h_eff  # explicit factor 2
    end

    if sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2]
        return 1 - _zero_dec(delta_E)
    else
        return _zero_dec(delta_E)
    end
end
