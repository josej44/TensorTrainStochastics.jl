# Generic tensor-train utilities.
# Convention (TensorTrains): the value of a TT `A` at `x` is A[1](x₁)⋯A[N](x_N) / A.z,
# so `z` is a normalization DENOMINATOR.

"""
    identity_tensor_train(N, qs)
    identity_tensor_train(N, qs...)
    identity_tensor_train(A::AbstractTensorTrain)

Tensor train of length `N` with physical dimensions `qs` that is identically equal to 1.
The third form copies length and physical dimensions from `A`.

# Example
```julia
I4 = identity_tensor_train(4, 2)      # 4 sites, Q = 2
evaluate(I4, [1, 2, 2, 1])            # 1.0
```
"""
identity_tensor_train(N, qs) = [ones(1, 1, qs...) for _ in 1:N] |> TensorTrain
identity_tensor_train(N, qs...) = identity_tensor_train(N, qs)
identity_tensor_train(A::AbstractTensorTrain) = identity_tensor_train(length(A), size(A[1])[3:end])

# Upper bound of max_x |B(x)|: product over sites of the entrywise maxima over x.
function estimate_norm_tt(B)
    B1 = (reshape(b, size(b, 1), size(b, 2), prod(size(b)[3:end])) for b in B)
    return (1 / abs(B.z)) * only(prod([maximum(abs(b[i, j, x]) for x in axes(b, 3)) for i in axes(b, 1), j in axes(b, 2)] for b in B1))
end

"""
    divide_by_constant!(B, constant)

Divide the tensor train `B` by `constant` in place (multiplies `B.z`). Returns `B`.

# Example
```julia
divide_by_constant!(B, 4)   # now B(x) is a quarter of what it was
```
"""
function divide_by_constant!(B, constant)
    B.z *= constant
    return B
end

"""
    multiply_by_constant!(B, constant)

Multiply the tensor train `B` by `constant` in place (divides `B.z`). Returns `B`.

# Example
```julia
two = multiply_by_constant!(identity_tensor_train(4, 2), 2)   # constant 2
```
"""
function multiply_by_constant!(B, constant)
    B.z /= constant
    return B
end

"""
    hadamard(A, B)

Pointwise (Hadamard) product of two tensor trains of the same type with matching
physical dimensions: `hadamard(A, B)(x) = A(x) * B(x)`. Bond dimensions multiply.

# Example
```julia
C = hadamard(A, B)
evaluate(C, x) ≈ evaluate(A, x) * evaluate(B, x)
```
"""
function hadamard(A::T, B::T) where T<:AbstractTensorTrain
    C = map(zip(A.tensors, B.tensors)) do (a, b)
        @assert size(a)[3:end] == size(b)[3:end]

        ar = reshape(a, size(a, 1), size(a, 2), prod(size(a)[3:end]))
        br = reshape(b, size(b, 1), size(b, 2), prod(size(b)[3:end]))

        @cast c[(ia, ib), (ja, jb), x] := ar[ia, ja, x] * br[ib, jb, x]

        reshape(c, size(c, 1), size(c, 2), size(a)[3:end]...)
    end

    T(C; z = A.z * B.z)
end

# Move the normalization `z` into the cores (value-preserving); afterwards `tt.z == 1`.
function absorb_z_into_matrices!(tt::AbstractTensorTrain)
    L = length(tt)

    if tt.z != 1 && tt.z != 0
        # Works for Logarithmic numbers too
        z_val = Float64(tt.z)
        factor = exp(log(abs(z_val)) / L)
        sign_factor = sign(z_val)

        for idx in 1:L
            if idx == 1
                tt[idx] *= sign_factor / factor
            else
                tt[idx] *= 1 / factor
            end
        end

        tt.z = one(typeof(tt.z))
    end
end

"""
    replica_product(A, B)

Joint tensor train of two independent replicas: `C((x, y)) = A(x) * B(y)`.

- 3-index cores (distributions): physical index of `C` is the combined index
  `x + Q*(y - 1)` (size `Q²`), i.e. for `Q = 2`: 1 = (1,1), 2 = (2,1), 3 = (1,2), 4 = (2,2).
  The swap `x ↔ y` is the permutation `[1, 3, 2, 4]`.
- 4-index cores (transition operators): `C((x,y), (x',y')) = A(x, x') * B(y, y')`,
  with the same combined-index convention on both physical indices.

# Example
```julia
P = replica_product(boltzmann_1, boltzmann_2)   # π₁ ⊗ π₂ on the joint state
```
"""
function replica_product(A::TensorTrain{F,3}, B::TensorTrain{G,3}) where {F,G}
    d = map(zip(A.tensors, B.tensors)) do (a, b)
        @tullio c[m1, m2, n1, n2, x, y] := a[m1, n1, x] * b[m2, n2, y]
        @cast _[(m1, m2), (n1, n2), (x, y)] := c[m1, m2, n1, n2, x, y]
    end
    return TensorTrain(d; z = A.z * B.z)
end

function replica_product(A::TensorTrain{F,4}, B::TensorTrain{G,4}) where {F,G}
    d = map(zip(A.tensors, B.tensors)) do (a, b)
        @tullio c[m1, m2, n1, n2, x, y, x1, y1] := a[m1, n1, x, x1] * b[m2, n2, y, y1]
        @cast _[(m1, m2), (n1, n2), (x, y), (x1, y1)] := c[m1, m2, n1, n2, x, y, x1, y1]
    end
    return TensorTrain(d; z = A.z * B.z)
end

"""
    compose_transitions(A, B)

Composition of two transition operators in tensor-train form (4-index cores
`[left, right, σ, σ_next]`): first `A`, then `B`,

    C(σ, σ'') = Σ_σ' A(σ, σ') B(σ', σ'') .

As matrices over configurations this is the product `W_A * W_B`.

# Example
```julia
A2 = compose_transitions(A, A)   # two steps
```
"""
function compose_transitions(A, B)
    d = map(zip(A.tensors, B.tensors)) do (a, b)
        @tullio c[m1, m2, n1, n2, x, x2] := a[m1, n1, x, x1] * b[m2, n2, x1, x2]
        @cast _[(m1, m2), (n1, n2), (x), (x2)] := c[m1, m2, n1, n2, x, x2]
    end
    return TensorTrain(d; z = A.z * B.z)
end

"""
    k_step_transition_tt(A::TensorTrain, steps::Int, bond::Int)

Transition operator for `steps` consecutive steps, `A^steps`, built by repeated
[`compose_transitions`](@ref) and compressed to bond dimension `bond` after each product.
For `steps == 1` the input `A` itself is returned.

# Example
```julia
A3 = k_step_transition_tt(A, 3, 16)
```
"""
function k_step_transition_tt(A::TensorTrain, steps::Int, bond::Int)
    A_step = A
    for _ in 1:steps-1
        A_step = compose_transitions(A_step, A)
        compress!(A_step; svd_trunc = TruncBond(bond))
    end
    return A_step
end

"""
    multi_step_transition_tt(A::TensorTrain, power_steps::Int, bond::Int)

Transition operator `A^(2^power_steps)` obtained by squaring `power_steps` times,
compressing to bond dimension `bond` after each squaring. The input is not modified.

# Example
```julia
A8 = multi_step_transition_tt(A, 3, 16)   # 8 steps
```
"""
function multi_step_transition_tt(A::TensorTrain, power_steps::Int, bond::Int)
    for _ in 1:power_steps
        A = compose_transitions(A, A)
        compress!(A; svd_trunc = TruncBond(bond))
    end
    return A
end

# Exact L2 norm of a tensor train, sqrt(Σₓ A(x)²), via left-orthogonalization.
# Gauge-independent and backward-stable: unlike sqrt(dot(A, A)) it has no √eps floor when
# A is a small difference such as 1 - B·X.
function _norm_l2(A)
    A = deepcopy(A)
    orthogonalize_left!(A; svd_trunc = TruncThresh(0.0))
    return norm(A[end]) / abs(float(A.z))
end

# Relative L2 residual ‖1 - B·X‖₂ / ‖1‖₂ of a pointwise inverse X of B.
# Pass B in canonical (left-orthogonal) form, see _canonical: in an ill-conditioned gauge the
# cancellation in 1 - B·X sets a floor of order eps·∏cond(gauge) on the computed residual.
_inverse_residual(B, X, Id) = _norm_l2(Id - hadamard(B, X)) / _norm_l2(Id)

# Value-preserving copy of A in left-orthogonal form (independent of the gauge of A).
function _canonical(A)
    A = deepcopy(A)
    orthogonalize_left!(A; svd_trunc = TruncThresh(0.0))
    return A
end

"""
    inverse_tt(B, bond; tol=1e-12, maxsteps=100, steps=nothing, verbose=false)

Pointwise inverse `X(x) ≈ 1 / B(x)` by the Newton–Schulz iteration
`X_{n+1} = X_n (2 - B X_n)` (products are Hadamard products), compressing to bond
dimension `bond` after every product. Starts from `X_0 = c (2 - c B)` with
`c = 1 / (2 · estimate of max|B|)`, which converges for `B > 0`.

By default the iteration stops when the relative L2 residual `‖1 - B·X‖₂ / ‖1‖₂` drops
below `tol`. The residual is computed exactly from the tensor trains, against a
left-orthogonalized copy of `B` (so it does not depend on the gauge of `B`) and by
orthogonalization rather than `dot` (which cannot resolve residuals below about 1e-8);
a warning is emitted if that does not happen within `maxsteps` iterations. Newton–Schulz
needs about `log₂(max B / min B)` iterations before it converges quadratically, so wide
dynamic ranges need more steps. Bond truncation sets a floor on the reachable residual.

`B` is rescaled in place by `normalize_eachmatrix!` (its value is unchanged).

# Keywords
- `tol::Real`: target relative L2 residual.
- `maxsteps::Int`: maximum number of iterations.
- `steps`: if an `Int`, run exactly that many iterations with no convergence check
  (the behaviour before the tolerance was introduced).
- `verbose::Bool`: print the starting constant, a progress bar and per-step diagnostics.

# Example
```julia
X = inverse_tt(B, 10)                 # until ‖1 - B·X‖₂/‖1‖₂ < 1e-12
X = inverse_tt(B, 10; steps = 20)     # exactly 20 iterations
```
"""
function inverse_tt(B, bond; tol::Real = 1e-12, maxsteps::Int = 100, steps = nothing,
                    verbose::Bool = false)
    normalize_eachmatrix!(B)
    B0 = 1 / (2 * estimate_norm_tt(B))
    verbose && @show B0

    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!(two - temp, B0)

    Id = identity_tensor_train(B)
    B_canonical = steps === nothing ? _canonical(B) : B     # only for the residual
    nmax = steps === nothing ? maxsteps : steps
    converged = steps !== nothing
    residual = NaN
    progress = Progress(nmax; enabled = verbose)
    for t in 1:nmax
        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = hadamard(B, Bn)
        normalize_eachmatrix!(temp1)
        compress!(temp1; svd_trunc = TruncBond(bond))

        Bnn = two - temp1

        Bn = hadamard(Bnn, Bn)

        normalize_eachmatrix!(Bn)
        compress!(Bn; svd_trunc = TruncBond(bond))

        verbose && @show estimate_norm_tt(Bn) Bn.z
        next!(progress)

        if steps === nothing
            residual = _inverse_residual(B_canonical, Bn, Id)
            verbose && @show t residual
            if residual < tol
                converged = true
                break
            end
        end
    end
    verbose && finish!(progress)
    converged || @warn "inverse_tt did not converge in $maxsteps steps: relative residual $residual ≥ tol = $tol"
    normalize_eachmatrix!(Bn)
    return Bn
end

# Newton variant with per-step `normalize!`; returns `(X, z_acc)`.
# Unused in the original code; kept unexported.
function inverse_tt_improve(B, bond; steps = 150, verbose::Bool = false)
    normalize_eachmatrix!(B)
    B0 = 1 / estimate_norm_tt(B)
    verbose && @show B0

    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!(two - temp, B0)

    z_ = 1.0
    z_acc = Logarithmic(1.0)
    progress = Progress(steps; enabled = verbose)
    for t in 1:steps
        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = hadamard(B, Bn)
        normalize_eachmatrix!(temp1)
        compress!(temp1; svd_trunc = TruncBond(bond))

        two_ = deepcopy(two)
        two_ = multiply_by_constant!(two_, z_)
        Bnn = two_ - temp1

        Bn = hadamard(Bnn, Bn)

        normalize_eachmatrix!(Bn)
        z_ = normalize!(Bn)
        compress!(Bn; svd_trunc = TruncBond(bond))
        z_acc *= z_^2

        verbose && @show estimate_norm_tt(Bn) Bn.z
        next!(progress)
    end
    normalize_eachmatrix!(Bn)
    return Bn, z_acc
end

"""
    inverse_secant_tt(B, bond; tol=1e-12, maxsteps=100, steps=nothing, verbose=false)

Pointwise inverse `X(x) ≈ 1 / B(x)` by the secant-type iteration
`X_{k+1} = X_k + X_{k-1} (I - B X_k)` (Hadamard products), whose residual satisfies
`e_{k+1} = e_k e_{k-1}`. Compresses to bond dimension `bond` after every operation.
Stopping rule and keywords as in [`inverse_tt`](@ref): by default iterate until the exact
relative L2 residual is below `tol` (warning after `maxsteps`); `steps = n` runs exactly
`n` iterations.

`B` is rescaled in place by `normalize_eachmatrix!` (its value is unchanged).

# Keywords
- `tol::Real`, `maxsteps::Int`, `steps`: see [`inverse_tt`](@ref).
- `verbose::Bool`: show a progress bar and print `‖I - B X‖` every 5 steps.

# Example
```julia
X = inverse_secant_tt(B, 10)
```
"""
function inverse_secant_tt(B, bond; tol::Real = 1e-12, maxsteps::Int = 100, steps = nothing,
                           verbose::Bool = false)
    # Normalize first
    normalize_eachmatrix!(B)

    # Initialization
    norm_B = estimate_norm_tt(B)
    Xminus1 = multiply_by_constant!(identity_tensor_train(B), 1.0 / (2.0 * norm_B))

    # Second approximation (one Newton step from Xminus1)
    Id = identity_tensor_train(B)
    BXm1 = hadamard(B, Xminus1)
    compress!(BXm1; svd_trunc = TruncBond(bond))
    two_Id = multiply_by_constant!(identity_tensor_train(B), 2.0)
    X = hadamard(Xminus1, two_Id - BXm1)
    normalize_eachmatrix!(X)
    compress!(X; svd_trunc = TruncBond(bond))

    B_canonical = steps === nothing ? _canonical(B) : B     # only for the residual
    nmax = steps === nothing ? maxsteps : steps
    converged = steps !== nothing
    residual = NaN
    progress = Progress(nmax; enabled = verbose)
    for t in 1:nmax
        # Secant: X_{k+1} = X_k + X_{k-1} * (I - B*X_k)

        # Step 1: B * X_k
        BX = hadamard(B, X)
        normalize_eachmatrix!(BX)
        compress!(BX; svd_trunc = TruncBond(bond))

        # Step 2: I - B*X_k
        residual = Id - BX
        normalize_eachmatrix!(residual)
        compress!(residual; svd_trunc = TruncBond(bond))

        # Step 3: X_{k-1} * (I - B*X_k)
        correction = hadamard(Xminus1, residual)
        normalize_eachmatrix!(correction)
        compress!(correction; svd_trunc = TruncBond(bond))

        # Step 4: X_{k+1} = X_k + correction
        Xplus1 = X + correction
        normalize_eachmatrix!(Xplus1)
        compress!(Xplus1; svd_trunc = TruncBond(bond))

        # Update for the next iteration
        Xminus1 = X
        X = Xplus1

        # Monitoring
        if verbose && t % 5 == 0
            error_est = estimate_norm_tt(Id - hadamard(B, X))
            println("  Step $t: ||I - B*X|| ≈ $error_est")
        end
        next!(progress)

        if steps === nothing
            residual = _inverse_residual(B_canonical, X, Id)
            if residual < tol
                converged = true
                break
            end
        end
    end
    verbose && finish!(progress)
    converged || @warn "inverse_secant_tt did not converge in $maxsteps steps: relative residual $residual ≥ tol = $tol"

    return X
end
