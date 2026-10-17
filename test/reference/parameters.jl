# ============================================
# PARAMETER STRUCT
# ============================================

"""
    MCParameters

Parameters of Monte Carlo simulations of Ising chains (mutable).

# System
- `N::Int`: number of spins
- `betas::Vector{Float64}`: inverse temperature of each chain
- `j_vector::Vector{Float64}`: nearest-neighbour couplings (length N-1)
- `h_vector::Vector{Float64}`: local fields (length N)

# Dynamics
- `p0::Float64`: probability of keeping the spin in parallel Glauber (default 0.0)
- `s::Float64`: fixed swap rate for parallel tempering (default 0.1)

# Simulation
- `seed::Int`: random seed (default 123)
"""
mutable struct MCParameters
    # System
    N::Int
    betas::Vector{Float64}
    j_vector::Vector{Float64}
    h_vector::Vector{Float64}

    # Dynamics
    p0::Float64
    s::Float64

    # Simulation
    seed::Int

    function MCParameters(
        N::Int,
        betas::Vector{Float64},
        j_vector::Vector{Float64},
        h_vector::Vector{Float64},
        p0::Float64,
        s::Float64,
        seed::Int
    )
        @assert N > 0 "N must be positive"
        @assert length(j_vector) == N - 1 "j_vector must have length N-1"
        @assert length(h_vector) == N "h_vector must have length N"
        @assert 0 <= p0 <= 1 "p0 must be in [0,1]"
        @assert s >= 0 "s must be non-negative"

        new(N, betas, j_vector, h_vector, p0, s, seed)
    end
end

"""
    MCParameters(; N, beta=nothing, betas=nothing, j_vector, h_vector, p0=0.0, s=0.1, seed=123)

Keyword constructor; accepts either `beta` (one chain) or `betas` (several chains).
"""
function MCParameters(;
    N::Int,
    beta::Union{Nothing, Real} = nothing,
    betas::Union{Nothing, Vector{<:Real}} = nothing,
    j_vector::Vector{<:Real},
    h_vector::Vector{<:Real},
    p0::Real = 0.0,
    s::Real = 0.1,
    seed::Int = 123
)
    @assert length(j_vector) == N - 1 "j_vector must have length N-1 = $(N-1), got $(length(j_vector))"
    @assert length(h_vector) == N "h_vector must have length N = $N, got $(length(h_vector))"

    if beta !== nothing && betas !== nothing
        error("give either 'beta' or 'betas', not both")
    elseif beta !== nothing
        betas_vec = [Float64(beta)]
    elseif betas !== nothing
        betas_vec = Float64.(betas)
    else
        error("you must give 'beta' or 'betas'")
    end

    return MCParameters(
        N,
        betas_vec,
        Float64.(j_vector),
        Float64.(h_vector),
        Float64(p0),
        Float64(s),
        seed
    )
end

"""
    MCParameters(m::ChainModel; betas=[m.β], p0=0.0, s=0.1, seed=123)

Adapter from the package's `ChainModel` (couplings and fields of `m`; `betas` gives one
inverse temperature per chain).
"""
MCParameters(m::ChainModel; betas = [m.β], p0 = 0.0, s = 0.1, seed::Int = 123) =
    MCParameters(; N = nsites(m), betas = Float64.(betas), j_vector = m.J, h_vector = m.h, p0, s, seed)

# ============================================
# UTILITIES
# ============================================

"Number of chains (length of `betas`)."
n_chains(params::MCParameters) = length(params.betas)

"Number of spins."
N(params::MCParameters) = params.N

"Check that the parameters are consistent."
function validate(params::MCParameters)
    @assert params.N > 0 "N must be positive"
    @assert length(params.j_vector) == params.N - 1 "j_vector must have length N-1"
    @assert length(params.h_vector) == params.N "h_vector must have length N"
    @assert 0 <= params.p0 <= 1 "p0 must be in [0,1]"
    @assert params.s >= 0 "s must be non-negative"
    return true
end

function Base.show(io::IO, params::MCParameters)
    n_ch = n_chains(params)
    println(io, "MCParameters:")
    println(io, "  N (spins): $(params.N)")
    println(io, "  chains: $n_ch")
    println(io, "  β: $(params.betas)")
    println(io, "  p₀: $(params.p0)")
    if n_ch > 1
        println(io, "  s (swap rate): $(params.s)")
    end
    println(io, "  seed: $(params.seed)")
end
