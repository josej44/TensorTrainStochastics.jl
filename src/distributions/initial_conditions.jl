# Initial product distributions P0 = [p₁, …, p_N] (one probability vector per site).

"""
    random_P0(N, Q=2) -> Vector{Vector{Float64}}

Random normalized initial product distribution for `N` sites with `Q` states each.

# Example
```julia
P0 = random_P0(5)       # 5 probability vectors of length 2
```
"""
function random_P0(N, Q = 2)
    P0 = [rand(Q) for _ in 1:N]
    for i in 1:N
        P0[i] /= sum(P0[i])  # normalize each probability vector
    end
    return P0
end

"""
    parallel_random_P0_fixed(N) -> Vector{Vector{Float64}}

Random initial product distribution for two replicas that start in the same
configuration: for every site the joint states (1,1) and (2,2) have random
probabilities and the mixed states (combined indices 2 and 3) have probability 0.

# Example
```julia
P0 = parallel_random_P0_fixed(5)   # 5 vectors of length 4
```
"""
function parallel_random_P0_fixed(N)
    P0 = [Float64[rand(), 0.0, 0.0, rand()] for _ in 1:N]
    for i in 1:N
        P0[i] ./= sum(P0[i])  # normalize each probability vector
    end
    return P0
end

"""
    transform_init_cond_simple(initial_probs) -> Vector{Vector{Float64}}

Product distribution for one chain from the probabilities `p_i = P(σᵢ = +1)`:
site `i` gets `[1 - p_i, p_i]`.

# Example
```julia
transform_init_cond_simple([0.2, 0.9])   # [[0.8, 0.2], [0.1, 0.9]]
```
"""
function transform_init_cond_simple(initial_probs)
    return [[1 - p, p] for p in initial_probs]
end

"""
    transform_init_cond_multiple(initial_probs) -> Vector{Vector{Float64}}

Product distribution for two replicas that start in the same configuration, from
`p_i = P(σᵢ = +1)`: site `i` gets `[1 - p_i, 0, 0, p_i]` on the combined index.

# Example
```julia
transform_init_cond_multiple([0.2])   # [[0.8, 0.0, 0.0, 0.2]]
```
"""
function transform_init_cond_multiple(initial_probs)
    return [[1 - p, 0.0, 0.0, p] for p in initial_probs]
end

# Product-state tensor train (bond dimension 1) of the distribution P0.
_product_tt(P0) = TensorTrain([(@tullio _[1, 1, x] := pi[x]) for pi in P0])
