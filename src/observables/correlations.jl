# Equal-time correlations of a normalized distribution tensor train.

"""
    correlation_next_pairs_simple(bt) -> Vector{Float64}

Nearest-neighbour moments `⟨σᵢ σᵢ₊₁⟩`, `i = 1, …, N-1`, of a single-chain normalized
distribution. Requires `N ≥ 3`.

# Example
```julia
c = correlation_next_pairs_simple(B)
```
"""
correlation_next_pairs_simple(bt) = [p[1] + p[4] - p[2] - p[3] for p in full_second_order_marginals_system(bt)]

"""
    second_moment_system(second_order_marginals_system) -> Vector{Tuple{Float64,Float64}}

Nearest-neighbour moments of both replicas from the joint pair marginals of a
two-replica distribution (4×4 matrices from [`full_second_order_marginals_system`](@ref)):
element `i` is `(⟨σᵢσᵢ₊₁⟩ of replica 1, ⟨σᵢσᵢ₊₁⟩ of replica 2)`.

# Example
```julia
sm = second_moment_system(full_second_order_marginals_system(B12))
```
"""
function second_moment_system(second_order_marginals_system)
    function sum_dif(matrix, chain_number)
        signs = if chain_number == 1
            # Pattern: sign alternates with the parity of i + j
            [(-1)^(i+j) for i in 1:4, j in 1:4]
        else  # chain_number == 2
            # Pattern: sign alternates with the parity of i÷2 + j÷2
            [(-1)^(div(i-1, 2) + div(j-1, 2)) for i in 1:4, j in 1:4]
        end
        return sum(signs .* matrix)
    end

    second_moments = Tuple{Float64, Float64}[]
    for k in 1:length(second_order_marginals_system)
        marginals = second_order_marginals_system[k]
        second_moment_ch_1 = sum_dif(marginals, 1)
        second_moment_ch_2 = sum_dif(marginals, 2)
        push!(second_moments, (second_moment_ch_1, second_moment_ch_2))
    end
    return second_moments
end

"""
    correlation_between_spins_system(second_moments_system, expected_values_system) -> Vector{Tuple{Float64,Float64}}

Pearson correlation of consecutive spins for both replicas,

    (⟨σᵢσᵢ₊₁⟩ - ⟨σᵢ⟩⟨σᵢ₊₁⟩) / sqrt(|1 - ⟨σᵢ⟩²| · |1 - ⟨σᵢ₊₁⟩²|) ,

from [`second_moment_system`](@ref) and [`marginal_ev_system`](@ref).

# Example
```julia
corr = correlation_between_spins_system(sm, ev)
```
"""
function correlation_between_spins_system(second_moments_system, expected_values_system)
    correlations = Tuple{Float64, Float64}[]
    N = length(second_moments_system) + 1
    for k in 1:N-1
        second_moment = second_moments_system[k]
        expected_value_1 = expected_values_system[k][1]
        expected_value_2 = expected_values_system[k+1][1]
        corr_ch_1 = (second_moment[1] - expected_value_1 * expected_value_2) / (sqrt(abs(1 - expected_value_1^2)) * sqrt(abs(1 - expected_value_2^2)))
        expected_value_1 = expected_values_system[k][2]
        expected_value_2 = expected_values_system[k+1][2]
        corr_ch_2 = (second_moment[2] - expected_value_1 * expected_value_2) / (sqrt(abs(1 - expected_value_1^2)) * sqrt(abs(1 - expected_value_2^2)))
        push!(correlations, (corr_ch_1, corr_ch_2))
    end
    return correlations
end

_potts2spin(x) = 2x - 3

_expectation(f, p::AbstractMatrix{<:Number}) = sum(f(xi) * f(xj) * p[xi, xj] for xi in axes(p, 1), xj in axes(p, 2); init = 0.0)
_expectation(f, p::AbstractVector{<:Number}) = sum(f(xi) * p[xi] for xi in eachindex(p); init = 0.0)

"""
    spatial_correlation(B, r; f=x -> 2x - 3) -> Float64

Spatially averaged moment at distance `r` of a single-chain distribution,

    C(r) = 1/(N - r) Σᵢ ⟨f(xᵢ) f(xᵢ₊ᵣ)⟩ ,   r = 1, …, N-1 ,

computed with TensorTrains' `twovar_marginals` (which normalizes every pair marginal,
so `B` need not be normalized).

# Example
```julia
C = [spatial_correlation(B, r) for r in 1:length(B)-1]
```
"""
function spatial_correlation(B, r; f = _potts2spin)
    N = length(B)
    margs_2 = twovar_marginals(B)    # matrix of P(xᵢ, xⱼ)

    C_r = 0.0
    for i in 1:(N - r)
        j = i + r
        Pij = margs_2[i, j]           # joint distribution P(xᵢ, xⱼ)
        C_r += _expectation(f, Pij)   # ⟨σᵢ σⱼ⟩ = Σ f(xᵢ) f(xⱼ) Pᵢⱼ
    end

    return C_r / (N - r)
end
