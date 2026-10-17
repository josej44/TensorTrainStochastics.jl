# Domain (cluster) statistics of a single-chain distribution.

"""
    portion_of_TT(A, i, j, index) -> Matrix

Product of the core matrices `A[i][:,:,index] ⋯ A[j][:,:,index]` (sites `i..j` all in
state `index`).

# Example
```julia
only(portion_of_TT(B, 1, length(B), 2))   # weight of the all-up configuration
```
"""
function portion_of_TT(A, i, j, index)
    matrix = A[i][:, :, index]
    for k in i+1:j
        matrix *= A[k][:, :, index]
    end
    return matrix
end

"""
    cluster_size_prob(B, l) -> Float64

Expected fraction of sites that belong to a domain (maximal run of equal spins) of
length exactly `l`, for the normalized single-chain distribution `B` (Q = 2):

    Σ_x P(x) · (number of sites in domains of length l) / N ,   1 ≤ l ≤ N .

Throws an `ArgumentError` if `l ∉ 1:N` or `Q ≠ 2`.

# Example
```julia
p = [cluster_size_prob(B, l) for l in 1:length(B)]   # sums to 1
```
"""
function cluster_size_prob(B, l)
    NN = length(B)
    1 <= l <= NN || throw(ArgumentError("domain length l must be in 1:$NN, got l = $l"))
    size(B[1], 3) == 2 || throw(ArgumentError("cluster_size_prob only supports Q = 2"))

    # Whole chain in one domain (bug C: this case used to throw a BoundsError)
    if l == NN
        return only(portion_of_TT(B, 1, NN, 1)) + only(portion_of_TT(B, 1, NN, 2))
    end

    left, right = full_subproducts_system(B)
    pushfirst!(left, ones(1, 1))
    push!(right, ones(1, 1))

    prob = 0.0

    for i in 1:NN-l+1
        if i == 1
            prob += (portion_of_TT(B, 1, l, 1) * B[l+1][:, :, 2] * right[l+1])[1]
            prob += (portion_of_TT(B, 1, l, 2) * B[l+1][:, :, 1] * right[l+1])[1]
        elseif i == NN - l + 1
            prob += (left[NN-l] * B[NN-l][:, :, 1] * portion_of_TT(B, NN-l+1, NN, 2))[1]
            prob += (left[NN-l] * B[NN-l][:, :, 2] * portion_of_TT(B, NN-l+1, NN, 1))[1]
        else
            prob += (left[i-1] * B[i-1][:, :, 2] * portion_of_TT(B, i, i+l-1, 1) * B[i+l][:, :, 2] * right[i+l])[1]
            prob += (left[i-1] * B[i-1][:, :, 1] * portion_of_TT(B, i, i+l-1, 2) * B[i+l][:, :, 1] * right[i+l])[1]
        end
    end
    return prob * l / NN
end
