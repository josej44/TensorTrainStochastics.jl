# Marginals and expected values of a normalized distribution tensor train.

"""
    full_marginals_system(B, full_subprod_system=1, full_system_already_compute=false)

Single-site marginals `[p₁, …, p_N]` of the normalized distribution `B` (for two
replicas, the marginals of the joint pair state at each site, length 4).
If `full_system_already_compute` is `true`, `full_subprod_system` must be the
`(left, right)` returned by [`full_subproducts_system`](@ref) and is reused.

# Example
```julia
normalize!(B)
p = full_marginals_system(B)      # p[i][q] = P(xᵢ = q)
```
"""
function full_marginals_system(B, full_subprod_system = 1, full_system_already_compute = false)

    if full_system_already_compute == false
        left, right = full_subproducts_system(B)
    else
        left, right = full_subprod_system
    end

    N = length(B.tensors)
    marginals = Vector{Float64}[]
    push!(marginals, [(B.tensors[1][:, :, q] * right[1])[1] for q in 1:size(B.tensors[1], 3)])
    for k in 2:N-1
        push!(marginals, [(left[k-1] * B.tensors[k][:, :, q] * right[k])[1] for q in 1:size(B.tensors[1], 3)])
    end
    push!(marginals, [(left[N-1] * B.tensors[N][:, :, q])[1] for q in 1:size(B.tensors[1], 3)])
    return marginals
end

"""
    full_simple_ev_system(B) -> Vector{Float64}

Magnetizations `⟨σᵢ⟩ = P(xᵢ = 2) - P(xᵢ = 1)` of a single-chain normalized distribution.

# Example
```julia
m = full_simple_ev_system(B)
```
"""
full_simple_ev_system(B) = [p[2] - p[1] for p in full_marginals_system(B)]

"""
    marginal_ev_system(marginal_system) -> Vector{Tuple{Float64,Float64}}

Magnetizations of both replicas from the joint single-site marginals (combined index
`x + 2(y - 1)`, as returned by [`full_marginals_system`](@ref) on a two-replica
distribution): element `i` is `(⟨σᵢ⟩ of replica 1, ⟨σᵢ⟩ of replica 2)`.
Each marginal must have length 4 (Q = 2), otherwise an `ArgumentError` is thrown.

# Example
```julia
ev = marginal_ev_system(full_marginals_system(B12))
```
"""
function marginal_ev_system(marginal_system)
    function expected(dist)
        length(dist) == 4 ||
            throw(ArgumentError("marginal_ev_system expects two-replica marginals of length 4 (Q = 2), got $(length(dist))"))
        return ((dist[1] + dist[3]) * (-1) + (dist[2] + dist[4]) * (1), (dist[1] + dist[2]) * (-1) + (dist[3] + dist[4]) * (1))
    end

    expected_values = Tuple{Float64, Float64}[]
    for marginal in marginal_system
        push!(expected_values, expected(marginal))
    end
    return expected_values
end

"""
    full_second_order_marginals_system(B, full_subprod_system=1, full_system_already_compute=false)

Joint marginals `P(xᵢ, xᵢ₊₁)` of every pair of consecutive sites of the normalized
distribution `B` (matrices of size `q × q`, `i = 1, …, N-1`). Arguments as in
[`full_marginals_system`](@ref). Requires `N ≥ 3` (an `ArgumentError` is thrown for `N = 2`).

# Example
```julia
P2 = full_second_order_marginals_system(B)   # P2[i][a, b] = P(xᵢ = a, xᵢ₊₁ = b)
```
"""
function full_second_order_marginals_system(B, full_subprod_system = 1, full_system_already_compute = false)
    N = length(B.tensors)
    N >= 3 || throw(ArgumentError("full_second_order_marginals_system needs N ≥ 3 sites, got N = $N"))

    if full_system_already_compute == false
        left, right = full_subproducts_system(B)
    else
        left, right = full_subprod_system
    end

    marginals = Matrix{Float64}[]

    # First marginal (k = 1)
    right_2 = right[2]
    @tullio corr_k[a, b] := B.tensors[1][i, j, a] * B.tensors[2][j, k, b] * right_2[k, m]
    push!(marginals, corr_k)

    # Middle marginals
    for k in 2:N-2
        left_k = left[k-1]
        right_k = right[k+1]
        @tullio corr_k_temp[a, b] := left_k[i, j] * B.tensors[k][j, m, a] * B.tensors[k+1][m, n, b] * right_k[n, o]
        push!(marginals, corr_k_temp)
    end

    # Last marginal
    left_end = left[N-2]
    @tullio corr_k_last[a, b] := left_end[i, j] * B.tensors[N-1][j, k, a] * B.tensors[N][k, m, b]
    push!(marginals, corr_k_last)

    return marginals
end
