# Random model instances.

"""
    random_chain_model(N, a=-1.0, b=1.0; rng=Random.default_rng(), β=rand(rng)) -> ChainModel

Random open chain of `N` sites with couplings `J` and fields `h` drawn uniformly
from `[a, b)`. Draw order: `β` (if not given), then `J`, then `h`.

# Arguments
- `N::Integer`: number of sites (`N ≥ 2`).
- `a::Real`, `b::Real`: bounds of the uniform distribution for `J` and `h`.

# Keywords
- `rng::AbstractRNG`: random number generator.
- `β::Real`: inverse temperature (default: uniform in `[0, 1)`).

# Example
```julia
using Random
m = random_chain_model(6; rng = MersenneTwister(1), β = 2.0)
```
"""
function random_chain_model(N::Integer, a::Real = -1.0, b::Real = 1.0;
                            rng::AbstractRNG = Random.default_rng(), β::Real = rand(rng))
    j_vector = a .+ (b - a) .* rand(rng, N - 1)     # couplings J_{i,i+1} (N-1 elements)
    h_vector = a .+ (b - a) .* rand(rng, N)         # local fields h_i (N elements)
    return ChainModel(j_vector, h_vector, β)
end
