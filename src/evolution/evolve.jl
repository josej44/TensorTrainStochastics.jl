# Time evolution of a distribution under a transition operator: B_{t+1}(σ') = Σ_σ B_t(σ) A(σ, σ').

"""
    distribution_b_tt(A, P0, t, bond, swapfun=identity, save=true; verbose=false)

Evolve the product initial distribution `P0` for `t` steps of the transition operator
`A` (4-index cores `[left, right, σ, σ_next]`). After every step the distribution is
compressed to bond dimension `bond`, passed through `swapfun` (e.g. [`ConstantSwap`](@ref),
[`EnergySwap`](@ref)), compressed again and normalized (`normalize!`).

# Arguments
- `A::TensorTrain`: transition operator.
- `P0::Vector{Vector{Float64}}`: one probability vector per site.
- `t::Int`: number of steps.
- `bond::Int`: maximum bond dimension.
- `swapfun`: function `B -> B'` applied after every step (default `identity`).
- `save::Bool`: return all distributions (`t + 1` of them, including `P0`) or only the last.

# Keywords
- `verbose::Bool`: show a progress bar.

# Example
```julia
m = ChainModel(ones(5), zeros(6), 1.0)
A = build_transition_tensortrain(m; update_rule = :sequential, bond = 8)
Bt = distribution_b_tt(A, random_P0(6), 200, 8)
```
"""
function distribution_b_tt(
    A::TensorTrain,
    P0::Vector{Vector{Float64}},
    t::Int,
    bond::Int,
    swapfun = identity,
    save::Bool = true;
    verbose::Bool = false
    )

    B = _product_tt(P0)
    lista_B_T = save ? [B] : nothing

    progress = Progress(t; enabled = verbose)
    for _ in 1:t
        # Time evolution
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1, m2, n1, n2, σ_next] := A_i[m1, n1, σ, σ_next] * B_i[m2, n2, σ]
            @cast _[(m1, m2), (n1, n2), σ_next] := new_[m1, m2, n1, n2, σ_next]
        end |> TensorTrain

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc = TruncBond(bond))

        B = swapfun(B)

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc = TruncBond(bond))
        normalize!(B)

        save && push!(lista_B_T, B)
        next!(progress)
    end

    return save ? lista_B_T : B
end

"""
    distribution_b_tt_exp(A, P0, t, bond, swapfun=identity, save=true; verbose=false)

Like [`distribution_b_tt`](@ref), but the operator is squared after every step
(`A ← A²`, compressed to `bond`). The `k`-th saved distribution has therefore evolved
for `2^k - 1` steps of the original `A` (0, 1, 3, 7, …), which reaches long times quickly.

# Example
```julia
Bt = distribution_b_tt_exp(A, random_P0(6), 10, 8)   # last one: 1023 steps
```
"""
function distribution_b_tt_exp(
    A::TensorTrain,
    P0::Vector{Vector{Float64}},
    t::Int,
    bond::Int,
    swapfun = identity,
    save::Bool = true;
    verbose::Bool = false
    )

    B = _product_tt(P0)
    lista_B_T = save ? [B] : nothing

    progress = Progress(t; enabled = verbose)
    for _ in 1:t
        # Time evolution
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1, m2, n1, n2, σ_next] := A_i[m1, n1, σ, σ_next] * B_i[m2, n2, σ]
            @cast _[(m1, m2), (n1, n2), σ_next] := new_[m1, m2, n1, n2, σ_next]
        end |> TensorTrain

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc = TruncBond(bond))

        B = swapfun(B)

        A = compose_transitions(A, A)
        compress!(A; svd_trunc = TruncBond(bond))

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc = TruncBond(bond))
        normalize!(B)

        save && push!(lista_B_T, B)
        next!(progress)
    end

    return save ? lista_B_T : B
end
