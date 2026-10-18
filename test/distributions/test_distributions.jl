const TTS = TensorTrainStochastics

@testset "boltzmann_tt = exp(-βE)" begin
    rng = StableRNG(11)
    for N in (2, 3, 5)
        m = random_chain_model(N; rng, β = 1.3)
        B = boltzmann_tt(m)
        @test length(B) == N
        @test all(tt_value(B, x) ≈ exp(-m.β * chain_energy(m, x)) for x in all_configs(N, 2))
        # beta_scale multiplies β; negative β is allowed
        B2 = boltzmann_tt(m; beta_scale = -0.5)
        @test all(tt_value(B2, x) ≈ exp(0.5 * m.β * chain_energy(m, x)) for x in all_configs(N, 2))
        # normalized: equals the exact distribution
        normalize!(B)
        @test [tt_value(B, x) for x in all_configs(N, 2)] ≈ boltzmann_exact(m)
    end
end

@testset "boltzmann_swap_tt and boltzmann_swap_n_tt" begin
    rng = StableRNG(12)
    for N in (2, 3, 4)
        m1 = random_chain_model(N; rng, β = 0.4)
        m2 = with_beta(m1, 1.7)
        R = boltzmann_swap_tt(m1, m2)
        R3 = boltzmann_swap_n_tt(m1, m2, 3)
        @test size(R[1], 3) == 4
        for x in all_configs(N, 2), y in all_configs(N, 2)
            idx = combined_index(x, y)
            r = exp((m1.β - m2.β) * (chain_energy(m1, x) - chain_energy(m1, y)))
            @test tt_value(R, idx) ≈ r                                         # definition
            @test tt_value(R, idx) ≈ energy_difference(m1, m2, collect(zip(x, y)))   # notebook check
            @test tt_value(R3, idx) ≈ energy_difference_n(m1, m2, collect(zip(x, y)), 3)
            @test tt_value(R3, idx) ≈ r^3
        end
    end
    m1 = random_chain_model(3; rng)
    @test_throws ArgumentError boltzmann_swap_tt(m1, random_chain_model(3; rng))
    @test_throws ArgumentError boltzmann_swap_n_tt(m1, random_chain_model(3; rng), 2)
end

@testset "initial conditions" begin
    for N in (2, 3, 6)
        P0 = random_P0(N)
        @test length(P0) == N && all(length.(P0) .== 2)
        @test all(p -> sum(p) ≈ 1 && all(>=(0), p), P0)
        @test all(length.(random_P0(N, 3)) .== 3)

        Pp = parallel_random_P0_fixed(N)
        @test all(p -> length(p) == 4 && p[2] == 0 && p[3] == 0 && sum(p) ≈ 1, Pp)
    end
    @test transform_init_cond_simple([0.2, 0.9]) ≈ [[0.8, 0.2], [0.1, 0.9]]
    @test transform_init_cond_multiple([0.2, 1.0]) ≈ [[0.8, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]]

    # product-state TT: value = Π p_i(x_i)
    for N in (2, 3)
        P0 = random_P0(N)
        B = TTS._product_tt(P0)
        @test all(tt_value(B, x) ≈ prod(P0[i][x[i]] for i in 1:N) for x in all_configs(N, 2))
    end
end
