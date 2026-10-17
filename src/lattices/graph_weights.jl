# Random weights on a graph structure.

"""
    add_gaussian_weights(adj::Matrix{Float64}; sJ=1.0, sh=0.5, rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Turn a binary adjacency matrix into a weighted model matrix `A`:
`A[i,j] = A[j,i] = J_ij ~ N(0, sJ²)` on every edge of `adj` and `A[i,i] = h_i ~ N(0, sh²)`.
Couplings are drawn in the order of the upper triangle (row by row), then the fields.

# Keywords
- `sJ::Float64`: standard deviation of the couplings.
- `sh::Float64`: standard deviation of the local fields.
- `rng`: random number generator.

# Example
```julia
A = add_gaussian_weights(reorder_rcm(random_3regular(20)); sJ = 0.5, sh = 0.2)
```
"""
function add_gaussian_weights(adj::Matrix{Float64};
                              sJ::Float64 = 1.0,
                              sh::Float64 = 0.5,
                              rng = Random.GLOBAL_RNG)
    N = size(adj, 1)
    A = zeros(N, N)

    # Upper triangle: the same J_ij at (i,j) and (j,i)
    for i in 1:N, j in i+1:N
        if adj[i, j] != 0.0
            Jij     = sJ * randn(rng)
            A[i, j] = Jij
            A[j, i] = Jij
        end
    end

    # Local field on the diagonal
    for i in 1:N
        A[i, i] = sh * randn(rng)
    end

    return A
end
