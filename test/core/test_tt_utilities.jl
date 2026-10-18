const TTS = TensorTrainStochastics

@testset "identity_tensor_train" begin
    for I1 in (identity_tensor_train(4, 2), identity_tensor_train(4, (2,)))
        @test length(I1) == 4
        @test all(tt_value(I1, x) == 1 for x in all_configs(4, 2))
    end
    A = rand_tt(StableRNG(1), [1, 2, 3, 1], 2, 3)       # 4-index cores, dims (2, 3)
    IA = identity_tensor_train(A)
    @test length(IA) == 3 && size(IA[2])[3:end] == (2, 3)
    @test tt_value(IA, [(1, 3), (2, 1), (2, 2)]) == 1
end

@testset "multiply/divide by constant" begin
    A = rand_tt(StableRNG(2), [1, 2, 2, 1], 2)
    x = [1, 2, 1]
    v = tt_value(A, x)
    @test multiply_by_constant!(A, 3.0) === A
    @test tt_value(A, x) ≈ 3v
    @test divide_by_constant!(A, 6.0) === A
    @test tt_value(A, x) ≈ v / 2
end

@testset "hadamard" begin
    rng = StableRNG(3)
    A = rand_tt(rng, [1, 2, 3, 1], 2); A.z = 7.0
    B = rand_tt(rng, [1, 3, 2, 1], 2); B.z = 0.2
    C = hadamard(A, B)
    @test all(tt_value(C, x) ≈ tt_value(A, x) * tt_value(B, x) for x in all_configs(3, 2))
    # 4-index cores
    A4 = rand_tt(rng, [1, 2, 2, 1], 2, 2)
    B4 = rand_tt(rng, [1, 2, 2, 1], 2, 2)
    C4 = hadamard(A4, B4)
    idx = [(1, 2), (2, 2), (2, 1)]
    @test tt_value(C4, idx) ≈ tt_value(A4, idx) * tt_value(B4, idx)
    # Base.* is not extended for tensor trains (no type piracy)
    @test !hasmethod(*, Tuple{typeof(A), typeof(A)})
end

@testset "replica_product" begin
    rng = StableRNG(4)
    Q, N = 2, 3
    A = rand_tt(rng, [1, 2, 2, 1], Q); A.z = 3.0
    B = rand_tt(rng, [1, 3, 2, 1], Q)
    C = replica_product(A, B)
    @test size(C[1], 3) == Q^2
    combined(x, y) = x + Q * (y - 1)
    @test all(tt_value(C, combined.(x, y)) ≈ tt_value(A, x) * tt_value(B, y)
              for x in all_configs(N, Q), y in all_configs(N, Q))
    # the swap permutation [1, 3, 2, 4] exchanges the replicas
    Cswap = TensorTrain([T[:, :, [1, 3, 2, 4]] for T in C.tensors]; z = C.z)
    Cba = replica_product(B, A)
    @test all(tt_value(Cswap, x) ≈ tt_value(Cba, x) for x in all_configs(N, Q^2))

    # 4-index cores: C((x,y),(x',y')) = A(x,x') B(y,y')
    A4 = rand_tt(rng, [1, 2, 2, 1], Q, Q)
    B4 = rand_tt(rng, [1, 2, 3, 1], Q, Q)
    C4 = replica_product(A4, B4)
    @test size(C4[2])[3:end] == (Q^2, Q^2)
    x, xp, y, yp = [1, 2, 2], [2, 2, 1], [2, 1, 1], [1, 1, 2]
    @test tt_value(C4, [(combined(x[i], y[i]), combined(xp[i], yp[i])) for i in 1:N]) ≈
          tt_value(A4, collect(zip(x, xp))) * tt_value(B4, collect(zip(y, yp)))
end

@testset "compose_transitions and powers" begin
    rng = StableRNG(5)
    Q, N = 2, 3
    A = rand_tt(rng, [1, 2, 2, 1], Q, Q)
    B = rand_tt(rng, [1, 2, 3, 1], Q, Q)
    WA, WB = dense_operator(A, Q), dense_operator(B, Q)
    @test dense_operator(compose_transitions(A, B), Q) ≈ WA * WB

    bond = 64                                                  # large enough: no truncation
    @test dense_operator(k_step_transition_tt(A, 3, bond), Q) ≈ WA^3
    @test k_step_transition_tt(A, 1, bond) === A
    Acopy = deepcopy(A)
    @test dense_operator(multi_step_transition_tt(A, 2, bond), Q) ≈ WA^4
    @test dense_operator(A, Q) ≈ dense_operator(Acopy, Q)        # input unchanged
end

@testset "sum of tensor trains with different z (TensorTrains ≥ 0.14)" begin
    rng = StableRNG(6)
    A = rand_tt(rng, [1, 3, 2, 3, 1], 2); A.z = 1000
    B = rand_tt(rng, [1, 2, 3, 2, 1], 2); B.z = 0.01
    @test all(tt_value(A + B, x) ≈ tt_value(A, x) + tt_value(B, x) for x in all_configs(4, 2))
    @test all(tt_value(A - B, x) ≈ tt_value(A, x) - tt_value(B, x) for x in all_configs(4, 2))
end

@testset "internal helpers" begin
    rng = StableRNG(7)
    A = rand_tt(rng, [1, 2, 3, 1], 2); A.z = 0.5
    @test TTS.estimate_norm_tt(A) >= maximum(abs(tt_value(A, x)) for x in all_configs(3, 2))
    vals = [tt_value(A, x) for x in all_configs(3, 2)]
    TTS.absorb_z_into_matrices!(A)
    @test A.z == 1
    @test [tt_value(A, x) for x in all_configs(3, 2)] ≈ vals
end

@testset "inverses" begin
    rng = StableRNG(8)
    N, Q = 4, 2
    # product-form B with values in [1, 1.3^4]: its inverse is exactly rank 1
    product_B() = TensorTrain([reshape(1 .+ 0.3 .* rand(rng, Q), 1, 1, Q) for _ in 1:N])
    B = product_B()
    vals = [tt_value(B, x) for x in all_configs(N, Q)]
    X = inverse_tt(B, 5; steps = 10)
    @test max_inverse_error(B, X) < 1e-10
    @test [tt_value(B, x) for x in all_configs(N, Q)] ≈ vals   # B rescaled, value unchanged

    # bug 2: inverse_secant_tt used to call the undefined `threshold_eraser`
    B2 = product_B()
    X2 = inverse_secant_tt(B2, 5; steps = 20)
    @test max_inverse_error(B2, X2) < 1e-10

    # a non-product B: 1 + 0.2·(random positive TT)
    R = rand_tt(rng, [1, 2, 2, 2, 1], Q)
    B3 = identity_tensor_train(R) + multiply_by_constant!(R, 0.2 / TTS.estimate_norm_tt(R))
    @test max_inverse_error(B3, inverse_tt(deepcopy(B3), 16; steps = 15)) < 1e-6
    @test max_inverse_error(B3, inverse_secant_tt(deepcopy(B3), 16; steps = 25)) < 1e-6

    # default: iterate until the relative L2 residual is below tol (bug I)
    @test max_inverse_error(B3, inverse_tt(deepcopy(B3), 16)) < 1e-10
    @test max_inverse_error(B3, inverse_secant_tt(deepcopy(B3), 16)) < 1e-10
    X3 = inverse_tt(deepcopy(B3), 16; steps = 1)                              # unconverged iterate
    exact_res = sqrt(sum(abs2, 1 .- tt_values(B3, Q) .* tt_values(X3, Q))) / sqrt(Q^N)
    @test TTS._inverse_residual(B3, X3, identity_tensor_train(B3)) ≈ exact_res rtol = 1e-8
    @test_logs (:warn, r"did not converge") inverse_tt(deepcopy(B3), 16; maxsteps = 2)
    @test_logs (:warn, r"did not converge") inverse_secant_tt(deepcopy(B3), 16; maxsteps = 2)

    # verbose = false (default) prints nothing; verbose = true does
    @test isempty(captured_output(() -> inverse_tt(product_B(), 5; steps = 3)))
    @test isempty(captured_output(() -> inverse_secant_tt(product_B(), 5; steps = 5)))
    @test !isempty(captured_output(() -> inverse_tt(product_B(), 5; steps = 3, verbose = true)))
end

# Insert G·G⁻¹ on every bond: same function, different gauge (and a looser estimate_norm_tt).
function random_gauge(B, rng; spread = 1.0)          # condition number 10^(2·spread) per bond
    T = deepcopy(B.tensors)
    for k in 1:length(T)-1
        d = size(T[k], 2)
        Qm = Matrix(qr(randn(rng, d, d)).Q)
        G = Qm * Diagonal(10 .^ range(-spread, spread; length = d)) * Qm'
        Gi = inv(G)
        T[k]   = stack([T[k][:, :, x] * G for x in axes(T[k], 3)])          # right bond × G
        T[k+1] = stack([Gi * T[k+1][:, :, x] for x in axes(T[k+1], 3)])     # G⁻¹ × left bond
    end
    return TensorTrain(T; z = B.z)
end

@testset "inverses are gauge-independent (bug I)" begin
    rng = StableRNG(9)
    N, Q = 5, 2
    R = rand_tt(rng, [1, 3, 3, 3, 3, 1], Q)
    B = identity_tensor_train(R) + multiply_by_constant!(R, 3.0 / TTS.estimate_norm_tt(R))
    Bg = random_gauge(B, rng)
    @test tt_values(Bg, Q) ≈ tt_values(B, Q)                                   # same function
    truemax = maximum(tt_values(B, Q))
    @test TTS.estimate_norm_tt(deepcopy(Bg)) > 1e3 * truemax                  # very loose start
    # the old fixed 10 steps fail in the loose gauge; the tolerance-based iteration does not
    @test max_inverse_error(Bg, inverse_tt(deepcopy(Bg), 16; steps = 10)) > 1e-3
    for inv_fun in (inverse_tt, inverse_secant_tt)
        @test max_inverse_error(B, (@test_logs inv_fun(deepcopy(B), 16))) < 1e-10    # converges, no warning
        @test max_inverse_error(Bg, (@test_logs inv_fun(deepcopy(Bg), 16))) < 1e-10
    end
end
