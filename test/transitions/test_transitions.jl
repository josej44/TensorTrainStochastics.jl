const RATES = (glauber_transition_rate, metropolis_transition_rate, temp_zero_transition_rate,
               (n, s, i, m) -> transition_rate_inertia(n, s, i, m; p0 = 0.3))
const RULES = Dict(:sequential => glauber_transition_rate, :metropolis => metropolis_transition_rate,
                   :zero_temp => temp_zero_transition_rate)

@testset "rates" begin
    rng = MersenneTwister(21)
    for N in (2, 3, 4)
        m = random_chain_model(N; rng, β = 0.8)
        for x in all_configs(N, 2), i in 1:N
            s = σmap.(x)
            nb = chain_neighbors(s, i)
            # each rate is a probability distribution over the new spin
            for rate in RATES
                @test rate(nb, 1, i, m) + rate(nb, -1, i, m) ≈ 1
                @test 0 <= rate(nb, 1, i, m) <= 1
            end
            h_eff = m.h[i] + (i > 1 ? m.J[i-1] * s[i-1] : 0.0) + (i < N ? m.J[i] * s[i+1] : 0.0)
            ΔE = 2 * s[i] * h_eff
            g = exp(m.β * h_eff) / (2cosh(m.β * h_eff))
            @test glauber_transition_rate(nb, 1, i, m) ≈ g
            @test transition_rate_inertia(nb, 1, i, m; p0 = 0.3) ≈ 0.7g + 0.3 * (s[i] == 1)
            @test transition_rate_inertia(nb, 1, i, m) ≈ g                    # p0 defaults to 0
            @test metropolis_transition_rate(nb, -s[i], i, m) ≈ min(1, exp(-m.β * ΔE))
            @test temp_zero_transition_rate(nb, -s[i], i, m) == (ΔE <= 0 ? 1 : 0)
        end
    end
end

@testset "chain builders = brute force (N = $N)" for N in (2, 3, 4)
    rng = MersenneTwister(22 + N)
    m = random_chain_model(N; rng, β = 1.1)
    for (rule, rate) in RULES
        W = dense_operator(build_transition_tensortrain(m; update_rule = rule, bond = 16), 2)
        @test W ≈ dense_sequential(rate, m)
        @test all(sum(W; dims = 2) .≈ 1)
    end
    Wp = dense_operator(build_transition_tensortrain(m; update_rule = :parallel, p0 = 0.2), 2)
    @test Wp ≈ dense_parallel((n, s, i, mm) -> transition_rate_inertia(n, s, i, mm; p0 = 0.2), m)
    @test all(sum(Wp; dims = 2) .≈ 1)
    # direct builder calls
    @test dense_operator(build_parallel_transition_tt(glauber_transition_rate, m), 2) ≈
          dense_parallel(glauber_transition_rate, m)
    @test dense_operator(build_sequential_transition_tt(metropolis_transition_rate, m, 16), 2) ≈
          dense_sequential(metropolis_transition_rate, m)
    # Boltzmann is stationary for sequential Glauber and Metropolis
    π = boltzmann_exact(m)
    for rule in (:sequential, :metropolis)
        W = dense_operator(build_transition_tensortrain(m; update_rule = rule, bond = 16), 2)
        @test π' * W ≈ π'
    end
end

@testset "builder argument errors" begin
    m = random_chain_model(3; rng = MersenneTwister(25))
    @test_throws ArgumentError build_sequential_transition_tt(glauber_transition_rate, m, 5, 3)
    @test_throws ArgumentError build_transition_tensortrain(m; update_rule = :heat_bath)
    @test_throws ArgumentError build_transition_tensortrain(GraphModel(m); update_rule = :sequential)
    @test_throws ArgumentError build_transition_tensortrain(GraphModel(m); Q = 3)
end

@testset "parallel_transition_tensor_train (N = $N)" for N in (2, 3)
    rng = MersenneTwister(26 + N)
    m1 = random_chain_model(N; rng, β = 0.5)
    m2 = with_beta(m1, 1.5)
    for rule in (:metropolis, :parallel)
        W1 = dense_operator(build_transition_tensortrain(m1; update_rule = rule, bond = 16), 2)
        W2 = dense_operator(build_transition_tensortrain(m2; update_rule = rule, bond = 16), 2)
        W12 = dense_operator(parallel_transition_tensor_train(m1, m2; update_rule = rule, bond = 16), 4)
        pos = Dict(c => k for (k, c) in enumerate(all_configs(N, 2)))
        expected = [begin
                        (x, y), (xp, yp) = split_replicas(c), split_replicas(cp)
                        W1[pos[x], pos[xp]] * W2[pos[y], pos[yp]]
                    end for c in all_configs(N, 4), cp in all_configs(N, 4)]
        @test W12 ≈ expected
    end
    @test_throws ArgumentError parallel_transition_tensor_train(m1, random_chain_model(N; rng);
                                                                 update_rule = :metropolis)
end

@testset "graph builder (N = $N)" for N in (2, 3, 4)
    rng = MersenneTwister(30 + N)
    # GraphModel(chain) reproduces the chain Metropolis operator exactly
    m = random_chain_model(N; rng, β = 0.9)
    Wg = dense_operator(build_transition_tensortrain(GraphModel(m); bond = 32), 2)
    @test Wg ≈ dense_operator(build_transition_tensortrain(m; update_rule = :metropolis, bond = 16), 2)
    @test Wg ≈ dense_graph_metropolis(GraphModel(m))
    # bond = 0: no compression
    @test dense_operator(build_transition_tensortrain(GraphModel(m); bond = 0), 2) ≈ Wg
end

@testset "graph builder on 3-regular graphs" begin
    for N in (4, 6)
        g = GraphModel(build_instance(N; seed = N), 0.7)
        W = dense_operator(build_transition_tensortrain(g; bond = 64), 2)
        @test W ≈ dense_graph_metropolis(g)
        @test all(sum(W; dims = 2) .≈ 1)
        π = boltzmann_exact(g)
        @test π' * W ≈ π'
    end
end

@testset "graph builder with an isolated node" begin
    # node 2 has no neighbours (complete graph minus its edges)
    A = [0.3 0.0 1.0; 0.0 -0.2 0.0; 1.0 0.0 0.1]
    g = GraphModel(A, 1.0)
    W = dense_operator(build_transition_tensortrain(g; bond = 32), 2)
    @test W ≈ dense_graph_metropolis(g)
    # bug H: a chain with a zero coupling at its end isolates site 1
    m = ChainModel([0.0, 0.7], [0.2, -0.1, 0.4], 1.2)
    @test dense_operator(build_transition_tensortrain(GraphModel(m); bond = 32), 2) ≈
          dense_operator(build_transition_tensortrain(m; update_rule = :metropolis, bond = 16), 2)
end
