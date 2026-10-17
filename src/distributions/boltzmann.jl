# Boltzmann distribution of a chain as a tensor train, and the replica swap ratio.

"""
    boltzmann_tt(m::ChainModel, Q=2, σ=x -> 2x - 3; beta_scale=1.0) -> TensorTrain

Unnormalized Boltzmann weight of the chain as a tensor train with bond dimension `Q`:

    B(x) = exp(-β' E(σ(x))),   β' = beta_scale · m.β ,

where `x[i] ∈ 1:Q` and `σ` maps indices to physical spins. Call `normalize!` on the
result to get the probability distribution.

# Arguments
- `m::ChainModel`: the chain.
- `Q::Int`: number of states per spin.
- `σ`: index → spin value map.

# Keywords
- `beta_scale::Float64`: multiplies `m.β` (ex keyword `coc`).

# Example
```julia
B = boltzmann_tt(ChainModel(ones(3), zeros(4), 1.0))
normalize!(B)
```
"""
function boltzmann_tt(m::ChainModel, Q::Int = 2, σ = x -> 2x - 3; beta_scale::Float64 = 1.0)
    N = nsites(m)

    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 3}[]

    h_vector = m.h
    j_vector = m.J
    beta = m.β * beta_scale

    # ============================================
    # First tensor A1
    # ============================================
    A1 = zeros(1, Q, Q)

    for sigma in 1:Q          # σ₁
        external_force = exp(beta * h_vector[1] * σ(sigma))
        # Row vector f₁(σ₁, ωⱼ) of size 1×Q; ωⱼ runs over the values of σ₂
        f_vector = zeros(1, Q)
        for j in 1:Q
            omega_j = σ(j)
            f_vector[1, j] = exp(beta * j_vector[1] * σ(sigma) * omega_j) * external_force
        end

        A1[1, :, sigma] = f_vector
    end

    push!(tensors, A1)

    # ============================================
    # Middle tensors A2, ..., A_{N-1}
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q, Q, Q)

        for sigma in 1:Q         # σᵢ
            external_force = exp(beta * h_vector[i] * σ(sigma))
            left_factor = delta_vector(σ(sigma))
            right_factor = zeros(1, Q)

            for j in 1:Q
                omega_j = σ(j)   # value of σᵢ₊₁
                right_factor[1, j] = exp(beta * j_vector[i] * σ(sigma) * omega_j) * external_force
            end

            Ai[:, :, sigma] = left_factor' * right_factor
        end

        push!(tensors, Ai)
    end

    # ============================================
    # Last tensor AN
    # ============================================
    AN = zeros(Q, 1, Q)

    for sigma in 1:Q
        external_force = exp(beta * h_vector[N] * σ(sigma))
        AN[:, 1, sigma] = delta_vector(σ(sigma))' * external_force
    end

    push!(tensors, AN)

    return TensorTrain(tensors)
end

"""
    boltzmann_swap_tt(m1::ChainModel, m2::ChainModel) -> TensorTrain

Replica swap ratio on the joint state `(x, y)` (replica 1 in `x` at `m1.β`, replica 2
in `y` at `m2.β`):

    r(x, y) = π(y, x) / π(x, y) = exp((β₁ - β₂)(E(x) - E(y))) ,

with the combined physical index `x + 2(y - 1)` of [`replica_product`](@ref).
The Metropolis swap acceptance is `min(1, r)`. `m1` and `m2` must have the same
couplings (only `β` may differ), otherwise an `ArgumentError` is thrown.

# Example
```julia
m1 = ChainModel(ones(3), zeros(4), 0.5); m2 = with_beta(m1, 2.0)
R = boltzmann_swap_tt(m1, m2)
```
"""
function boltzmann_swap_tt(m1::ChainModel, m2::ChainModel)
    _check_replicas(m1, m2)
    model_1 = with_beta(m1, m2.β - m1.β)
    model_2 = with_beta(m1, m1.β - m2.β)
    return replica_product(boltzmann_tt(model_1), boltzmann_tt(model_2))
end

"""
    boltzmann_swap_n_tt(m1::ChainModel, m2::ChainModel, n) -> TensorTrain

`n`-th power of the replica swap ratio, `r(x, y)^n` (see [`boltzmann_swap_tt`](@ref)).

# Example
```julia
R3 = boltzmann_swap_n_tt(m1, m2, 3)
```
"""
function boltzmann_swap_n_tt(m1::ChainModel, m2::ChainModel, n)
    _check_replicas(m1, m2)
    model_1 = with_beta(m1, n * (m2.β - m1.β))
    model_2 = with_beta(m1, n * (m1.β - m2.β))
    return replica_product(boltzmann_tt(model_1), boltzmann_tt(model_2))
end
