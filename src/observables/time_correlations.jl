# Two-time observables from the joint distribution P(σ₀, σₜ) as a 4-index tensor train
# (cores [left, right, σ₀, σₜ]).

"""
    full_tcorr_system(B) -> (moments, correlations)

From the normalized joint distribution `B(σ₀, σₜ)` (4-index cores), the two-time moments
`⟨σᵢ(0) σᵢ(t)⟩` and the connected correlations `⟨σᵢ(0) σᵢ(t)⟩ - ⟨σᵢ(0)⟩⟨σᵢ(t)⟩` of every
site. Only `Q = 2` is supported.

# Example
```julia
moments, corrs = full_tcorr_system(Bt[end])
```
"""
function full_tcorr_system(B)
    (size(B.tensors[1], 3) == 2 && size(B.tensors[1], 4) == 2) ||
        throw(ArgumentError("full_tcorr_system only supports Q = 2"))

    left, right = full_subproducts_system(B)

    function cov_(B, i)
        return B.tensors[i][:, :, 1, 1] + B.tensors[i][:, :, 2, 2] - B.tensors[i][:, :, 1, 2] - B.tensors[i][:, :, 2, 1]
    end

    function magn_(B, i)
        return B.tensors[i][:, :, 1, 2] + B.tensors[i][:, :, 2, 2] - B.tensors[i][:, :, 1, 1] - B.tensors[i][:, :, 2, 1]
    end

    function magn_0(B, i)
        return B.tensors[i][:, :, 2, 1] + B.tensors[i][:, :, 2, 2] - B.tensors[i][:, :, 1, 2] - B.tensors[i][:, :, 1, 1]
    end

    N = length(B.tensors)
    moments_ = Float64[]
    correlations_ = Float64[]

    aux_moment = only(cov_(B, 1) * right[1])
    push!(moments_, aux_moment)
    push!(correlations_, aux_moment - only(magn_(B, 1) * right[1]) * only(magn_0(B, 1) * right[1]))

    for k in 2:N-1
        aux_moment = only(left[k-1] * cov_(B, k) * right[k])
        push!(moments_, aux_moment)
        push!(correlations_, aux_moment - only(left[k-1] * magn_0(B, k) * right[k]) * only(left[k-1] * magn_(B, k) * right[k]))
    end

    aux_moment = only(left[N-1] * cov_(B, N))
    push!(moments_, aux_moment)
    push!(correlations_, aux_moment - only(left[N-1] * magn_0(B, N)) * only(left[N-1] * magn_(B, N)))

    return moments_, correlations_
end

"""
    times_correlation_evolution(A, B0, t, bond; verbose=false) -> (list_B_t, list_of_moments, list_of_correlations)

Evolve the joint distribution `P(σ₀, σₛ)` under the transition operator `A`, starting from
the distribution `B0` at time 0 (a 3-index `TensorTrain`, e.g. a normalized Boltzmann TT,
or a product distribution `P0::Vector{Vector{Float64}}`). Returns, for lags `s = 1, …, t`,
the joint TTs (normalized, compressed to `bond`) and the outputs of [`full_tcorr_system`](@ref).

# Keywords
- `verbose::Bool`: show a progress bar.

# Example
```julia
B = boltzmann_tt(m); normalize!(B)
Bs, moments, corrs = times_correlation_evolution(A, B, 100, 10)
```
"""
function times_correlation_evolution(
    A::TensorTrain,
    B::TensorTrain,
    t::Int,
    bond::Int;
    verbose::Bool = false
    )

    # Lag 1: lift B to the joint (σ₀, σ₁)
    B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1, m2, n1, n2, σ, σ_next] := A_i[m1, n1, σ, σ_next] * B_i[m2, n2, σ]
            @cast _[(m1, m2), (n1, n2), σ, σ_next] := new_[m1, m2, n1, n2, σ, σ_next]
    end |> TensorTrain

    normalize!(B)

    list_B_t = [B]

    covs_, corrs_ = full_tcorr_system(B)
    list_of_moments = [covs_]
    list_of_correlations = [corrs_]

    progress = Progress(t; enabled = verbose)
    next!(progress)
    for _ in 2:t
        # Time evolution of σₜ (σ₀ is carried along)
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1, m2, n1, n2, σ_0, σ_next] := A_i[m1, n1, σ, σ_next] * B_i[m2, n2, σ_0, σ]
            @cast _[(m1, m2), (n1, n2), σ_0, σ_next] := new_[m1, m2, n1, n2, σ_0, σ_next]
        end |> TensorTrain

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc = TruncBond(bond))

        normalize!(B)

        push!(list_B_t, B)

        covs_, corrs_ = full_tcorr_system(B)

        push!(list_of_moments, covs_)
        push!(list_of_correlations, corrs_)
        next!(progress)
    end

    return list_B_t, list_of_moments, list_of_correlations
end

times_correlation_evolution(A::TensorTrain, P0::Vector{Vector{Float64}}, t::Int, bond::Int; verbose::Bool = false) =
    times_correlation_evolution(A, _product_tt(P0), t, bond; verbose)
