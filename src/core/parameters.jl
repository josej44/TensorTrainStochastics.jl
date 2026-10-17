# Spin models: the physical system (couplings, fields, inverse temperature).
# Dynamics parameters (e.g. the inertia p0) are keywords of the rate functions,
# not fields of the model.

"""
    AbstractSpinModel

Supertype of the Ising-type spin models. Every subtype carries its couplings,
its local fields and an inverse temperature `β`, and implements [`nsites`](@ref),
[`local_fields`](@ref) and [`with_beta`](@ref).
"""
abstract type AbstractSpinModel end

"""
    ChainModel(J, h, β)

Open one-dimensional Ising chain with energy

    E(σ) = -Σᵢ J[i] σᵢ σᵢ₊₁ - Σᵢ h[i] σᵢ ,   σᵢ ∈ {-1, +1}.

# Arguments
- `J::AbstractVector{<:Real}`: nearest-neighbour couplings, `J[i]` couples sites `i` and `i+1`
  (length `N-1`).
- `h::AbstractVector{<:Real}`: local fields (length `N`).
- `β::Real`: inverse temperature. Any real number except `NaN` is accepted; negative values
  are needed internally (e.g. the replica swap ratio uses `β₂ - β₁`).

The vectors are copied, so later changes to the inputs do not affect the model.
Throws an `ArgumentError` if `N < 2`, `length(J) != N - 1` or `β` is `NaN`.

# Example
```julia
m = ChainModel(ones(4), zeros(5), 1.0)   # homogeneous chain, N = 5
nsites(m)                                # 5
```
"""
struct ChainModel <: AbstractSpinModel
    J::Vector{Float64}
    h::Vector{Float64}
    β::Float64
    function ChainModel(J::Vector{Float64}, h::Vector{Float64}, β::Float64)
        N = length(h)
        N >= 2 || throw(ArgumentError("a chain needs at least 2 sites, got N = $N"))
        length(J) == N - 1 ||
            throw(ArgumentError("length(J) must be N-1 = $(N - 1), got $(length(J))"))
        isnan(β) && throw(ArgumentError("β must not be NaN"))
        return new(copy(J), copy(h), β)
    end
end

ChainModel(J::AbstractVector{<:Real}, h::AbstractVector{<:Real}, β::Real) =
    ChainModel(convert(Vector{Float64}, J), convert(Vector{Float64}, h), Float64(β))

"""
    GraphModel(A, β)
    GraphModel(m::ChainModel)

Ising model on an arbitrary graph, encoded in a symmetric matrix `A`:
`A[i,j] = J_ij` for `i ≠ j` (`0` means no edge) and `A[i,i] = h_i`. The energy is

    E(σ) = -Σ_{i<j} A[i,j] σᵢ σⱼ - Σᵢ A[i,i] σᵢ .

# Arguments
- `A::AbstractMatrix{<:Real}`: square, symmetric, at least 2×2.
- `β::Real`: inverse temperature (any real except `NaN`).

The matrix is copied. Throws an `ArgumentError` if `A` is not square, smaller than 2×2,
not symmetric, or if `β` is `NaN`.

`GraphModel(m::ChainModel)` builds the equivalent tridiagonal model (same `β`).

# Example
```julia
g = GraphModel([0.1 1.0; 1.0 -0.2], 0.5)   # two coupled spins
GraphModel(ChainModel([1.0, 2.0], zeros(3), 1.0)).A
```
"""
struct GraphModel <: AbstractSpinModel
    A::Matrix{Float64}
    β::Float64
    function GraphModel(A::Matrix{Float64}, β::Float64)
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square, got size $(size(A))"))
        size(A, 1) >= 2 || throw(ArgumentError("a graph needs at least 2 sites, got N = $(size(A, 1))"))
        issymmetric(A) || throw(ArgumentError("A must be symmetric"))
        isnan(β) && throw(ArgumentError("β must not be NaN"))
        return new(copy(A), β)
    end
end

GraphModel(A::AbstractMatrix{<:Real}, β::Real) = GraphModel(convert(Matrix{Float64}, A), Float64(β))

function GraphModel(m::ChainModel)
    A = diagm(0 => m.h)
    for i in eachindex(m.J)
        A[i, i+1] = m.J[i]
        A[i+1, i] = m.J[i]
    end
    return GraphModel(A, m.β)
end

"""
    nsites(m::AbstractSpinModel) -> Int

Number of spins `N` of the model.

# Example
```julia
nsites(ChainModel(ones(2), zeros(3), 1.0))   # 3
```
"""
nsites(m::ChainModel) = length(m.h)
nsites(m::GraphModel) = size(m.A, 1)

"""
    local_fields(m::AbstractSpinModel) -> Vector{Float64}

Local fields `h` of the model (for a `GraphModel`, the diagonal of `A`).
For a `ChainModel` the stored vector itself is returned; do not mutate it.

# Example
```julia
local_fields(GraphModel([0.1 1.0; 1.0 -0.2], 1.0))   # [0.1, -0.2]
```
"""
local_fields(m::ChainModel) = m.h
local_fields(m::GraphModel) = diag(m.A)

"""
    with_beta(m::AbstractSpinModel, β::Real)

Copy of `m` with the same couplings and fields and inverse temperature `β`.
Used to build replicas (two models that differ only in `β`).

# Example
```julia
m1 = ChainModel(ones(4), zeros(5), 0.5)
m2 = with_beta(m1, 2.0)
```
"""
with_beta(m::ChainModel, β::Real) = ChainModel(m.J, m.h, β)
with_beta(m::GraphModel, β::Real) = GraphModel(m.A, β)
