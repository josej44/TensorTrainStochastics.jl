const TTS = TensorTrainStochastics

# exact π₁ ⊗ π₂ on the joint state (combined index) and the swap ratio r(x, y)
function replica_setup(N; β1 = 0.4, β2 = 0.9, seed = 40)
    m1 = random_chain_model(N; rng = MersenneTwister(seed + N), β = β1)
    m2 = with_beta(m1, β2)
    B1 = boltzmann_tt(m1); normalize!(B1)
    B2 = boltzmann_tt(m2); normalize!(B2)
    Π = replica_product(B1, B2)
    r(c) = begin
        x, y = split_replicas(c)
        exp((m1.β - m2.β) * (chain_energy(m1, x) - chain_energy(m1, y)))
    end
    return m1, m2, Π, r
end

swapped(c) = [(k == 2 ? 3 : k == 3 ? 2 : k) for k in c]

@testset "EnergySwap (bug B): invariance of π₁ ⊗ π₂ (N = $N)" for N in (2, 3)
    m1, m2, Π, r = replica_setup(N)
    configs = all_configs(N, 4)
    πvals = tt_values(Π, 4)

    # exact Metropolis acceptance min(1, r)
    a_exact = tt_from_function(c -> min(1.0, r(c)), N, 4)
    @test all(tt_value(a_exact, c) ≈ min(1.0, r(c)) for c in configs)
    P1 = EnergySwap(a_exact)(deepcopy(Π))
    @test sum(tt_values(P1, 4)) ≈ 1                     # mass conserved (no normalize!)
    @test tt_values(P1, 4) ≈ πvals                      # π₁ ⊗ π₂ invariant

    # from an arbitrary distribution: mass conserved, equals the exact swap kernel
    Pr = TTS._product_tt([normalize(rand(MersenneTwister(N), 4), 1) for _ in 1:N])
    pr = tt_values(Pr, 4)
    pos = Dict(c => k for (k, c) in enumerate(configs))
    expected = [pr[k] * (1 - min(1, r(c))) + pr[pos[swapped(c)]] * min(1, r(swapped(c)))
                for (k, c) in enumerate(configs)]
    Pr1 = EnergySwap(a_exact)(deepcopy(Pr))
    @test tt_values(Pr1, 4) ≈ expected
    @test sum(tt_values(Pr1, 4)) ≈ 1

    # tt_swap approximation: close to (r+…+rⁿ)/(1+…+rⁿ), and invariance up to its accuracy
    n = 3
    a_tt = tt_swap(m1, m2, n, 32)
    f(c) = (rc = r(c); sum(rc^k for k in 1:n) / sum(rc^k for k in 0:n))
    @test maximum(abs(tt_value(a_tt, c) - f(c)) for c in configs) < 1e-6
    P2 = EnergySwap(a_tt)(deepcopy(Π))
    @test sum(tt_values(P2, 4)) ≈ 1 atol = 1e-6
    @test maximum(abs.(tt_values(P2, 4) .- πvals)) < 1e-6
    # steps is passed to inverse_tt: one Newton step is visibly less accurate
    a_1 = tt_swap(m1, m2, n, 32; steps = 1)
    @test maximum(abs(tt_value(a_1, c) - f(c)) for c in configs) > 1e-3
end

@testset "ConstantSwap and no_swap" begin
    N = 3
    _, _, Π, _ = replica_setup(N)
    Pr = TTS._product_tt([normalize(rand(MersenneTwister(1), 4), 1) for _ in 1:N])
    pr = tt_values(Pr, 4)
    pos = Dict(c => k for (k, c) in enumerate(all_configs(N, 4)))
    expected = [0.75 * pr[k] + 0.25 * pr[pos[swapped(c)]] for (k, c) in enumerate(all_configs(N, 4))]
    @test tt_values(ConstantSwap(0.25)(deepcopy(Pr)), 4) ≈ expected
    @test sum(tt_values(ConstantSwap(0.25)(deepcopy(Pr)), 4)) ≈ 1
    @test no_swap()(Pr) === Pr
end

@testset "distribution_b_tt = dense evolution (N = $N)" for N in (2, 3)
    m = random_chain_model(N; rng = MersenneTwister(50 + N), β = 0.8)
    P0 = random_P0(N)
    p0 = tt_values(TTS._product_tt(P0), 2)
    for rule in (:sequential, :metropolis, :zero_temp, :parallel)
        A = build_transition_tensortrain(m; update_rule = rule, bond = 16)
        W = dense_operator(A, 2)
        Bt = distribution_b_tt(A, P0, 6, 16)                    # bug 1: default swapfun works
        @test length(Bt) == 7
        @test all(tt_values(Bt[k+1], 2) ≈ vec(p0' * W^k) for k in 0:6)
        @test tt_values(distribution_b_tt(A, P0, 6, 16, identity, false), 2) ≈ vec(p0' * W^6)
    end
    # distribution_b_tt_exp: k-th entry evolved 2^k - 1 steps
    A = build_transition_tensortrain(m; update_rule = :metropolis, bond = 16)
    W = dense_operator(A, 2)
    Bexp = distribution_b_tt_exp(A, P0, 4, 16)
    @test all(tt_values(Bexp[k+1], 2) ≈ vec(p0' * W^(2^k - 1)) for k in 0:4)
end

@testset "long-time sequential dynamics converge to Boltzmann" begin
    for N in (2, 3), rule in (:sequential, :metropolis)
        m = random_chain_model(N; rng = MersenneTwister(60 + N), β = 0.7)
        A = build_transition_tensortrain(m; update_rule = rule, bond = 16)
        B = distribution_b_tt_exp(A, random_P0(N), 14, 16, identity, false)   # 2^14 - 1 steps
        @test tt_values(B, 2) ≈ boltzmann_exact(m) atol = 1e-8
    end
end

@testset "two replicas with swap = dense evolution" begin
    N = 2
    m1, m2, _, r = replica_setup(N)
    A12 = parallel_transition_tensor_train(m1, m2; update_rule = :metropolis, bond = 16)
    W12 = dense_operator(A12, 4)
    configs = all_configs(N, 4)
    pos = Dict(c => k for (k, c) in enumerate(configs))
    Pswap = zeros(length(configs), length(configs))
    for (k, c) in enumerate(configs)
        Pswap[k, pos[swapped(c)]] = 1
    end
    P0 = parallel_random_P0_fixed(N)
    p = tt_values(TTS._product_tt(P0), 4)

    S = 0.75I + 0.25Pswap                                     # ConstantSwap(0.25)
    Bt = distribution_b_tt(A12, P0, 4, 32, ConstantSwap(0.25))
    pk = copy(p)
    for k in 1:4
        pk = vec((pk' * W12) * S)
        @test tt_values(Bt[k+1], 4) ≈ pk
    end

    a_exact = tt_from_function(c -> min(1.0, r(c)), N, 4)    # EnergySwap with exact acceptance
    K = [(k == l ? 1 - min(1, r(c)) : 0.0) + (l == pos[swapped(c)] ? min(1, r(c)) : 0.0)
         for (k, c) in enumerate(configs), l in eachindex(configs)]
    Bt = distribution_b_tt(A12, P0, 4, 32, EnergySwap(a_exact))
    pk = copy(p)
    for k in 1:4
        pk = vec((pk' * W12) * K)
        @test tt_values(Bt[k+1], 4) ≈ pk
    end
end

@testset "verbose" begin
    m = random_chain_model(3; rng = MersenneTwister(70))
    A = build_transition_tensortrain(m; update_rule = :sequential, bond = 8)
    @test isempty(captured_output(() -> distribution_b_tt(A, random_P0(3), 3, 8)))
    m1 = random_chain_model(2; rng = MersenneTwister(71), β = 0.4)
    @test isempty(captured_output(() -> tt_swap(m1, with_beta(m1, 0.6), 2, 8)))
    @test !isempty(captured_output(() -> tt_swap(m1, with_beta(m1, 0.6), 2, 8; steps = 2, verbose = true)))
    # ProgressMeter only prints after its update interval, so a short run may print nothing
    @test length(distribution_b_tt(A, random_P0(3), 3, 8; verbose = true)) == 4
end
