# Replica-exchange (swap) steps acting on a two-replica distribution B(x, y)
# (combined physical index x + 2(y - 1); the swap x ↔ y is the permutation [1, 3, 2, 4]).
# Each constructor returns a function B -> B' to be passed as `swapfun` to distribution_b_tt.

# Tensor train with the replicas exchanged: B_swap(x, y) = B(y, x).
function _swap_replicas(B)
    swap_idx = [1, 3, 2, 4]
    tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
    return TensorTrain(tensors_swap; z = B.z)
end

"""
    ConstantSwap(swap::Real)

Swap step with a fixed probability: `B' = (1 - swap) B + swap B_swap`, where
`B_swap(x, y) = B(y, x)`. Returns a function `B -> B'` (the input's `z` is rescaled).

# Example
```julia
Bt = distribution_b_tt(A12, P0, 100, 10, ConstantSwap(0.1))
```
"""
function ConstantSwap(swap::Real)

    function (B)
        B_swap = _swap_replicas(B)
        multiply_by_constant!(B_swap, swap)
        multiply_by_constant!(B, (1 - swap))
        return B + B_swap
    end
end

"""
    EnergySwap(swap::TensorTrain)

Swap step with a configuration-dependent acceptance `a(x, y)` (a tensor train on the
joint state, e.g. [`tt_swap`](@ref) for Metropolis acceptance `≈ min(1, r)`):

    B'(x, y) = B(x, y) (1 - a(x, y)) + B(y, x) a(y, x) .

This conserves total probability and, for an acceptance satisfying detailed balance
with respect to `π₁ ⊗ π₂`, leaves `π₁ ⊗ π₂` invariant. Returns a function `B -> B'`.

# Example
```julia
a = tt_swap(m1, m2, 3, 10)
Bt = distribution_b_tt(A12, P0, 100, 10, EnergySwap(a))
```
"""
function EnergySwap(swap::TensorTrain)
    # Acceptance of the move out of the swapped state: a_swap(x, y) = a(y, x)
    swap_back = _swap_replicas(swap)
    function (B)
        B_swap = _swap_replicas(B)
        hadamard(B, identity_tensor_train(swap) - swap) + hadamard(B_swap, swap_back)
    end
end

"""
    no_swap()

Swap step that does nothing (returns `B` unchanged).

# Example
```julia
Bt = distribution_b_tt(A12, P0, 100, 10, no_swap())
```
"""
function no_swap()
    function (B)
        return B
    end
end

"""
    tt_swap(m1::ChainModel, m2::ChainModel, n, bond; steps=10, verbose=false) -> TensorTrain

Smooth tensor-train approximation of the Metropolis swap acceptance `min(1, r)`:

    a(x, y) = (r + r² + … + rⁿ) / (1 + r + … + rⁿ) ,   r = exp((β₁ - β₂)(E(x) - E(y))) ,

which tends to `min(1, r)` as `n → ∞` and satisfies detailed balance for every `n`.
The denominator `1 + r + … + rⁿ` is inverted with [`inverse_tt`](@ref); `bond` bounds all
compressions.

**Accuracy.** Newton–Schulz needs roughly `log₂(max/min)` steps to converge at the smallest
values of the denominator, so with the default `steps = 10` the result degrades when `r`
(hence `1 + r + … + rⁿ`) spans more than about 3 orders of magnitude, i.e. for large
`|β₁ - β₂| · ΔE` or large `n`. Increase `steps` in that regime and check the result.

# Keywords
- `steps::Int`: Newton iterations passed to [`inverse_tt`](@ref).
- `verbose::Bool`: passed to [`inverse_tt`](@ref).

# Example
```julia
m1 = ChainModel(ones(3), zeros(4), 0.5)
a = tt_swap(m1, with_beta(m1, 0.8), 3, 10)
```
"""
function tt_swap(m1::ChainModel, m2::ChainModel, n, bond; steps::Int = 10, verbose::Bool = false)
    tt = boltzmann_swap_tt(m1, m2)    # r

    for i in 2:n
        tt_i = boltzmann_swap_n_tt(m1, m2, i)       # rⁱ
        tt += tt_i                                  # r + r² + ... + rⁱ
        normalize_eachmatrix!(tt)
        compress!(tt; svd_trunc = TruncBond(bond))
    end

    tt_plus = tt + identity_tensor_train(tt)        # 1 + r + ... + rⁿ
    normalize_eachmatrix!(tt_plus)

    tt_inverse = inverse_tt(tt_plus, bond; steps, verbose)   # (1 + r + ... + rⁿ)^(-1)
    return hadamard(tt, tt_inverse)                 # (r + ... + rⁿ) / (1 + r + ... + rⁿ)
end
