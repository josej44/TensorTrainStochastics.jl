# Partial sums of a distribution tensor train, shared by the observables.
# The observables assume a NORMALIZED distribution (`normalize!`, so z = 1 and Σ B = 1).

"""
    full_subproducts_system(B) -> (left, right)

Partial contractions of the distribution `B` with every physical index summed:
`left[k] = Σ B¹⋯Bᵏ` (a row vector) and `right[k] = Σ Bᵏ⁺¹⋯Bᴺ` (a column vector),
for `k = 1, …, N-1`. Works for 3-index cores (one physical index per site) and for
4-index cores (two physical indices per site, e.g. the joint (σ₀, σₜ) of
[`times_correlation_evolution`](@ref)).

# Example
```julia
left, right = full_subproducts_system(B)
only(left[1] * right[1])          # total mass (1 for a normalized B)
```
"""
function full_subproducts_system(B::TensorTrain{F,3}) where F
    N = length(B.tensors)
    @tullio init_left[a, b] := B.tensors[1][a, b, l]
    @tullio init_right[a, b] := B.tensors[N][a, b, l]
    left = [init_left]
    right = [init_right]
    for k in 2:N-1
        @tullio next_left[a, b] := B.tensors[k][a, b, l]
        init_left = init_left * next_left
        push!(left, init_left)
        @tullio next_right[a, b] := B.tensors[N+1-k][a, b, l]
        init_right = next_right * init_right
        push!(right, init_right)
    end
    right = reverse(right)
    return left, right
end

function full_subproducts_system(B::TensorTrain{F,4}) where F
    N = length(B.tensors)
    @tullio init_left[a, b] := B.tensors[1][a, b, l, ll]
    @tullio init_right[a, b] := B.tensors[N][a, b, l, ll]
    left = [init_left]
    right = [init_right]
    for k in 2:N-1
        @tullio next_left[a, b] := B.tensors[k][a, b, l, ll]
        init_left = init_left * next_left
        push!(left, init_left)
        @tullio next_right[a, b] := B.tensors[N+1-k][a, b, l, ll]
        init_right = next_right * init_right
        push!(right, init_right)
    end
    right = reverse(right)
    return left, right
end
