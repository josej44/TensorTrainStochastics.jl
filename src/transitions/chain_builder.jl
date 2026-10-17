# Transition operators of an open chain as tensor trains (MPOs).
# Cores are indexed [left, right, σᵗ, σᵗ⁺¹]; for every σᵗ the values sum to 1 over σᵗ⁺¹.

"""
    build_parallel_transition_tt(transition_rate, model::ChainModel, Q=2, σ=x -> 2x - 3) -> TensorTrain

Transition operator of a parallel update (all sites at once):
`A(x, x') = Πᵢ transition_rate(neighbours of i in x, σ(x'ᵢ), i, model)`. Bond dimension `Q²`.

# Arguments
- `transition_rate`: a rate with the signature of [`glauber_transition_rate`](@ref).
- `model::ChainModel`: the chain.
- `Q::Int`, `σ`: number of states per spin and index → spin map.

# Example
```julia
A = build_parallel_transition_tt(glauber_transition_rate, ChainModel(ones(3), zeros(4), 1.0))
```
"""
function build_parallel_transition_tt(transition_rate, model::ChainModel, Q::Int = 2, σ = x -> 2x - 3)
    N = nsites(model)  # number of sites in the chain

    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 4}[]

    # ============================================
    # SITE 1: A₁[1, Q², σ₁ᵗ, σ₁ᵗ⁺¹]
    # Form: (f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ))ⱼ · (I ⊗ v_σ₁ᵗ)
    # f₁ is a 1×Q row vector, v_σ₁ᵗ is a 1×Q row vector,
    # (I ⊗ v_σ₁ᵗ) is Q×Q², result: (1×Q) · (Q×Q²) = 1×Q²
    # ============================================
    A1 = zeros(1, Q^2, Q, Q)

    for sigma_t in 1:Q          # σ₁ᵗ
        for sigma_t_plus in 1:Q  # σ₁ᵗ⁺¹

            # Row vector f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ) of size 1×Q; ωⱼ runs over the values of σ₂ᵗ
            f_vector = zeros(1, Q)
            for j in 1:Q
                omega_j = σ(j)
                neighbors = [σ(sigma_t), omega_j]  # [σ₁ᵗ, σ₂ᵗ]
                f_vector[1, j] = transition_rate(neighbors, σ(sigma_t_plus), 1, model)
            end

            v_sigma = delta_vector(σ(sigma_t))
            I_kron_v = kron(Matrix(I, Q, Q), v_sigma)  # Q×Q²

            # f₁ · (I ⊗ v_σ₁ᵗ) = (1×Q) · (Q×Q²) = 1×Q²
            A1[1, :, sigma_t, sigma_t_plus] = f_vector * I_kron_v
        end
    end

    push!(tensors, A1)

    # ============================================
    # MIDDLE SITES: Aᵢ[Q², Q², σᵢᵗ, σᵢᵗ⁺¹]
    # Form: (v_σᵢᵗᵀ ⊗ I) · Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) · (I ⊗ v_σᵢᵗ)
    # (Q²×Q) · (Q×Q) · (Q×Q²) = Q²×Q²
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q^2, Q^2, Q, Q)

        for sigma_t in 1:Q         # σᵢᵗ
            for sigma_t_plus in 1:Q # σᵢᵗ⁺¹

                # F_i[k,l] = fᵢ(ωₖ, σᵢᵗ, σᵢᵗ⁺¹, ωₗ) with ωₖ = σᵢ₋₁ᵗ and ωₗ = σᵢ₊₁ᵗ
                F_matrix = zeros(Q, Q)

                for k in 1:Q  # ωₖ = σᵢ₋₁ᵗ (left neighbour)
                    for l in 1:Q  # ωₗ = σᵢ₊₁ᵗ (right neighbour)
                        omega_k = σ(k)
                        omega_l = σ(l)
                        neighbors = [omega_k, σ(sigma_t), omega_l]  # [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
                        F_matrix[k, l] = transition_rate(neighbors, σ(sigma_t_plus), i, model)
                    end
                end

                v_sigma = delta_vector(σ(sigma_t))
                v_sigma_T = transpose(v_sigma)
                vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))   # Q²×Q
                I_kron_v = kron(Matrix(I, Q, Q), v_sigma)      # Q×Q²

                Ai[:, :, sigma_t, sigma_t_plus] = vT_kron_I * F_matrix * I_kron_v
            end
        end

        push!(tensors, Ai)
    end

    # ============================================
    # SITE N: Aₙ[Q², 1, σₙᵗ, σₙᵗ⁺¹]
    # Form: (v_σₙᵗᵀ ⊗ I) · (fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ))ⱼ,  (Q²×Q) · (Q×1) = Q²×1
    # ============================================
    AN = zeros(Q^2, 1, Q, Q)

    for sigma_t in 1:Q         # σₙᵗ
        for sigma_t_plus in 1:Q # σₙᵗ⁺¹

            # Column vector fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ) of size Q×1; ωⱼ runs over the values of σₙ₋₁ᵗ
            f_vector = zeros(Q, 1)
            for j in 1:Q
                omega_j = σ(j)
                neighbors = [omega_j, σ(sigma_t)]  # [σₙ₋₁ᵗ, σₙᵗ]
                f_vector[j, 1] = transition_rate(neighbors, σ(sigma_t_plus), N, model)
            end

            v_sigma = delta_vector(σ(sigma_t))
            v_sigma_T = transpose(v_sigma)
            vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))   # Q²×Q

            AN[:, 1, sigma_t, sigma_t_plus] = vT_kron_I * f_vector
        end
    end

    push!(tensors, AN)

    return TensorTrain(tensors)
end

"""
    build_sequential_transition_tt(transition_rate, model::ChainModel, bond=5, Q=2, σ=x -> 2x - 3) -> TensorTrain

Transition operator of one sequential step (a random site is updated):
`A = (1/N) Σᵢ Wᵢ`, where `Wᵢ` updates site `i` with `transition_rate` and leaves the
others unchanged. The terms are summed one by one, compressing to bond dimension
`bond` before and after each sum.

Only `Q = 2` is supported (the identity cores are hard-coded); other values throw an
`ArgumentError`.

# Example
```julia
A = build_sequential_transition_tt(metropolis_transition_rate, ChainModel(ones(3), zeros(4), 1.0), 8)
```
"""
function build_sequential_transition_tt(transition_rate, model::ChainModel, bond = 5, Q::Int = 2, σ = x -> 2x - 3)
    Q == 2 || throw(ArgumentError("build_sequential_transition_tt only supports Q = 2, got Q = $Q"))
    N = nsites(model)

    function transition_event(i, model, transition_rate)
        if i == 1
            tt_i = Array{Float64, 4}[]
            A1 = zeros(1, Q, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    f_vector = zeros(1, Q)
                    for j in 1:Q
                        omega_j = σ(j)
                        neighbors = [σ(sigma), omega_j]
                        f_vector[1, j] = transition_rate(neighbors, σ(sigma_plus), 1, model)
                    end
                    A1[1, :, sigma, sigma_plus] = f_vector
                end
            end
            push!(tt_i, A1)
            push!(tt_i, [1 ; 0 ;;; 0 ; 0 ;;;; 0 ; 0 ;;; 0 ; 1])
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]), 3:N)

        elseif i == N
            tt_i = Array{Float64, 4}[]
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]), 1:N-2)
            push!(tt_i, [1  0;;; 0 0 ;;;; 0 0 ;;; 0 1])
            AN = zeros(Q, 1, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    f_vector = zeros(Q, 1)
                    for j in 1:Q
                        omega_j = σ(j)
                        neighbors = [omega_j, σ(sigma)]
                        f_vector[j, 1] = transition_rate(neighbors, σ(sigma_plus), N, model)
                    end
                    AN[:, 1, sigma, sigma_plus] = f_vector
                end
            end
            push!(tt_i, AN)

        else
            tt_i = Array{Float64, 4}[]
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]), 1:i-2)
            push!(tt_i, [1  0;;; 0 0 ;;;;0 0 ;;; 0 1])

            Ai = zeros(Q, Q, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    F_matrix = zeros(Q, Q)
                    for k in 1:Q
                        for l in 1:Q
                            omega_k = σ(k)
                            omega_l = σ(l)
                            neighbors = [omega_k, σ(sigma), omega_l]
                            F_matrix[k, l] = transition_rate(neighbors, σ(sigma_plus), i, model)
                        end
                    end
                    Ai[:, :, sigma, sigma_plus] = F_matrix
                end
            end
            push!(tt_i, Ai)
            push!(tt_i, [1 ; 0 ;;; 0 ; 0 ;;;; 0 ; 0 ;;; 0 ; 1])
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]), i+2:N)
        end
        return TensorTrain(tt_i)
    end

    tensors = transition_event(1, model, transition_rate)

    # Compress the first event right away
    compress!(tensors, svd_trunc = TruncBond(bond))

    for i in 2:N
        tt_i = transition_event(i, model, transition_rate)

        # Compress the new event BEFORE adding it
        compress!(tt_i, svd_trunc = TruncBond(bond))

        tensors += tt_i

        # Compress after each sum
        compress!(tensors, svd_trunc = TruncBond(bond))
    end

    divide_by_constant!(tensors, N)
    return tensors
end

"""
    build_transition_tensortrain(model::ChainModel; update_rule, bond=5, Q=2, σ=x -> 2x - 3, p0=0.0) -> TensorTrain
    build_transition_tensortrain(model::GraphModel; update_rule=:metropolis, bond=10, Q=2, σ=s -> 2s - 3) -> TensorTrain

Transition operator of one step of the chosen dynamics.

For a `ChainModel`, `update_rule` is one of
- `:sequential`: Glauber, one random site per step (sequential builder);
- `:metropolis`: Metropolis, one random site per step;
- `:zero_temp`: zero-temperature dynamics, one random site per step;
- `:parallel`: Glauber with inertia `p0`, all sites at once (parallel builder).

For a `GraphModel` only `:metropolis` (sequential) is available.
Any other value throws an `ArgumentError`.

# Keywords
- `update_rule::Symbol`: the dynamics.
- `bond::Int`: maximum bond dimension for the compressions (graph: `bond = 0` means no compression).
- `Q::Int`, `σ`: number of states per spin and index → spin map.
- `p0::Real`: inertia of the parallel Glauber rule (chain only).

# Example
```julia
m = ChainModel(ones(5), zeros(6), 1.0)
A = build_transition_tensortrain(m; update_rule = :sequential, bond = 8)
```
"""
function build_transition_tensortrain(
    model::ChainModel;
    update_rule::Symbol,
    bond::Int = 5,
    Q::Int = 2,
    σ::Function = x -> 2x - 3,
    p0::Real = 0.0
    )
    if update_rule == :sequential
        return build_sequential_transition_tt(glauber_transition_rate, model, bond, Q, σ)
    elseif update_rule == :parallel
        rate = (nbrs, s_new, i, m) -> transition_rate_inertia(nbrs, s_new, i, m; p0)
        return build_parallel_transition_tt(rate, model, Q, σ)
    elseif update_rule == :metropolis
        return build_sequential_transition_tt(metropolis_transition_rate, model, bond, Q, σ)
    elseif update_rule == :zero_temp
        return build_sequential_transition_tt(temp_zero_transition_rate, model, bond, Q, σ)
    else
        throw(ArgumentError("unknown update rule: $update_rule. Use :sequential, :parallel, :metropolis or :zero_temp."))
    end
end

"""
    parallel_transition_tensor_train(m1::ChainModel, m2::ChainModel; update_rule, bond=5, Q=2, σ=x -> 2x - 3, p0=0.0) -> TensorTrain

Transition operator of two independent replicas (`m1` and `m2`, same couplings,
different `β`) on the joint state, `A((x,y), (x',y')) = A₁(x, x') A₂(y, y')`, with the
combined index of [`replica_product`](@ref). Keywords as in
[`build_transition_tensortrain`](@ref).

# Example
```julia
m1 = ChainModel(ones(3), zeros(4), 0.5)
A12 = parallel_transition_tensor_train(m1, with_beta(m1, 2.0); update_rule = :metropolis)
```
"""
function parallel_transition_tensor_train(m1::ChainModel, m2::ChainModel; update_rule, bond = 5, Q = 2,
                                          σ = x -> 2x - 3, p0 = 0.0)
    _check_replicas(m1, m2)
    A1 = build_transition_tensortrain(m1; update_rule, bond, Q, σ, p0)
    A2 = build_transition_tensortrain(m2; update_rule, bond, Q, σ, p0)
    return replica_product(A1, A2)
end
