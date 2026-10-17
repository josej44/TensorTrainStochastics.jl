# Random graph instances for GraphModel.
# Pipeline: random_3regular -> reorder_rcm -> add_gaussian_weights (see build_instance).

"""
    random_3regular(N; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Random 3-regular graph on `N` nodes (configuration model with rejection, at most
1000 attempts), returned as a binary symmetric adjacency matrix with zero diagonal
(`adj[i,j] = 1` if there is an edge).

Throws an `ArgumentError` unless `N` is even and `N ≥ 4`.

# Example
```julia
using Random
adj = random_3regular(10; rng = MersenneTwister(42))
```
"""
function random_3regular(N::Int; rng = Random.GLOBAL_RNG)
    (iseven(N) && N >= 4) ||
        throw(ArgumentError("a 3-regular graph needs an even N ≥ 4, got N = $N"))

    for _ in 1:1000
        stubs = repeat(1:N, inner = 3)
        shuffle!(rng, stubs)

        adj = zeros(N, N)
        ok  = true
        for k in 1:2:length(stubs)-1
            u, v = stubs[k], stubs[k+1]
            if u == v || adj[u, v] != 0
                ok = false; break
            end
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        end

        ok && return adj
    end
    error("could not generate a 3-regular graph for N = $N")
end

"""
    reorder_rcm(adj::Matrix{Float64}) -> Matrix{Float64}

Renumber the nodes of the graph `adj` (binary adjacency matrix) with the reverse
Cuthill–McKee ordering, which reduces the bandwidth and hence the bond dimensions of
the transition tensor train. Returns the reordered binary adjacency matrix (same
edges, new labels). Disconnected nodes are appended at the end of the ordering.

# Example
```julia
adj = reorder_rcm(random_3regular(20))
```
"""
function reorder_rcm(adj::Matrix{Float64})
    N = size(adj, 1)

    # Adjacency lists from the matrix
    neighbors = [[j for j in 1:N if adj[i, j] != 0.0] for i in 1:N]

    # BFS returning the last node reached from `start`
    function bfs_farthest(start)
        dist = fill(-1, N)
        dist[start] = 0
        q = [start]
        last = start
        for u in q, v in neighbors[u]
            if dist[v] == -1
                dist[v] = dist[u] + 1
                push!(q, v)
                last = v
            end
        end
        return last
    end

    # Start node: an end of the pseudo-diameter
    start = bfs_farthest(bfs_farthest(1))

    # Cuthill-McKee: BFS visiting neighbours by increasing degree
    visited = falses(N)
    order   = Int[]
    queue   = [start]
    visited[start] = true

    while !isempty(queue)
        u = popfirst!(queue)
        push!(order, u)
        unvisited = filter(v -> !visited[v], neighbors[u])
        for v in sort(unvisited, by = v -> length(neighbors[v]))
            visited[v] = true
            push!(queue, v)
        end
    end

    # Append disconnected nodes, if any
    for u in 1:N
        visited[u] && continue
        visited[u] = true
        push!(order, u)
    end

    # Reverse (RCM): better bandwidth than plain CM
    perm = reverse(order)

    # inv_perm[original_node] = new_position
    inv_perm = zeros(Int, N)
    for (new_pos, old_node) in enumerate(perm)
        inv_perm[old_node] = new_pos
    end

    # Reordered adjacency matrix: new_adj[π(i), π(j)] = adj[i, j]
    new_adj = zeros(N, N)
    for i in 1:N, j in 1:N
        if adj[i, j] != 0.0
            new_adj[inv_perm[i], inv_perm[j]] = 1.0
        end
    end

    return new_adj
end

"""
    build_instance(N; sJ=1.0, sh=0.5, seed=42) -> Matrix{Float64}

Full pipeline: random 3-regular graph ([`random_3regular`](@ref)), RCM reordering
([`reorder_rcm`](@ref)) and Gaussian weights ([`add_gaussian_weights`](@ref)), all
driven by `MersenneTwister(seed)`. Returns the matrix `A` (`A[i,j] = J_ij`,
`A[i,i] = h_i`), ready for `GraphModel(A, β)`.

# Keywords
- `sJ::Float64`: standard deviation of the couplings.
- `sh::Float64`: standard deviation of the local fields.
- `seed::Int`: random seed.

# Example
```julia
m = GraphModel(build_instance(20; sJ = 0.5, sh = 0.2), 1.0)
```
"""
function build_instance(N::Int;
                        sJ::Float64 = 1.0,
                        sh::Float64 = 0.5,
                        seed::Int   = 42)
    rng = MersenneTwister(seed)
    adj = random_3regular(N; rng)
    adj = reorder_rcm(adj)
    A   = add_gaussian_weights(adj; sJ, sh, rng)
    return A
end
