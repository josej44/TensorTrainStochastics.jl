const TTS = TensorTrainStochastics

normalized_boltzmann(m) = (B = boltzmann_tt(m); normalize!(B); B)

# exact single-chain expectations from the dense distribution
function exact_single(m)
    N = nsites(m); configs = all_configs(N, 2); p = boltzmann_exact(m)
    s(x) = σmap.(x)
    marg = [[sum(p[k] for (k, x) in enumerate(configs) if x[i] == q) for q in 1:2] for i in 1:N]
    pair = [[sum(p[k] for (k, x) in enumerate(configs) if x[i] == a && x[i+1] == b) for a in 1:2, b in 1:2] for i in 1:N-1]
    mag = [sum(p[k] * s(x)[i] for (k, x) in enumerate(configs)) for i in 1:N]
    nn = [sum(p[k] * s(x)[i] * s(x)[i+1] for (k, x) in enumerate(configs)) for i in 1:N-1]
    E = sum(p[k] * chain_energy(m, x) for (k, x) in enumerate(configs))
    return (; marg, pair, mag, nn, E, p, configs)
end

@testset "single-chain observables (N = $N)" for N in (2, 3, 4)
    m = random_chain_model(N; rng = MersenneTwister(80 + N), β = 0.9)
    B = normalized_boltzmann(m)
    ex = exact_single(m)

    left, right = full_subproducts_system(B)
    @test length(left) == length(right) == N - 1
    @test all(only(left[k] * right[k]) ≈ 1 for k in 1:N-1)

    @test full_marginals_system(B) ≈ ex.marg
    @test full_marginals_system(B, (left, right), true) ≈ ex.marg
    @test full_marginals_system(B) ≈ marginals(B)                     # TensorTrains cross-check
    @test full_simple_ev_system(B) ≈ ex.mag

    if N == 2
        @test_throws ArgumentError full_second_order_marginals_system(B)
        @test_throws ArgumentError correlation_next_pairs_simple(B)
        @test_throws ArgumentError energy_function_simple(B, m)
    else
        P2 = full_second_order_marginals_system(B)
        @test P2 ≈ ex.pair
        tv = twovar_marginals(B)
        @test all(P2[i] ≈ tv[i, i+1] for i in 1:N-1)                   # TensorTrains cross-check
        @test correlation_next_pairs_simple(B) ≈ ex.nn
        @test energy_function_simple(B, m) ≈ ex.E
        obs = system_description_over_time_simple([B, B], m)
        @test obs.full_expected_values_t[2] ≈ ex.mag && obs.energy_t[1] ≈ ex.E
    end

    # spatial correlation C(r) = mean_i ⟨σᵢσᵢ₊ᵣ⟩
    for r in 1:N-1
        Cr = sum(sum(ex.p[k] * σmap(x[i]) * σmap(x[i+r]) for (k, x) in enumerate(ex.configs)) for i in 1:N-r) / (N - r)
        @test spatial_correlation(B, r) ≈ Cr
        @test spatial_correlation(boltzmann_tt(m), r) ≈ Cr                # works unnormalized
    end
end

@testset "min_energy_config (N = $N)" for N in (2, 3, 5)
    m = random_chain_model(N; rng = MersenneTwister(90 + N))
    config, spins, E0 = min_energy_config(m)
    energies = [chain_energy(m, x) for x in all_configs(N, 2)]
    @test E0 ≈ minimum(energies)
    @test chain_energy(m, config) ≈ E0
    @test spins == σmap.(config)
end

@testset "two-replica observables (N = $N)" for N in (2, 3, 4)
    m1 = random_chain_model(N; rng = MersenneTwister(100 + N), β = 0.5)
    m2 = with_beta(m1, 1.4)
    # a correlated two-replica distribution: π₁⊗π₂ after one fixed-rate swap
    B12 = ConstantSwap(0.3)(replica_product(normalized_boltzmann(m1), normalized_boltzmann(m2)))
    normalize!(B12)
    configs = all_configs(N, 4)
    p = tt_values(B12, 4)
    xs = [split_replicas(c) for c in configs]
    expv(f) = sum(p[k] * f(xs[k]...) for k in eachindex(configs))

    marg = full_marginals_system(B12)
    ev = marginal_ev_system(marg)
    @test all(ev[i][1] ≈ expv((x, y) -> σmap(x[i])) && ev[i][2] ≈ expv((x, y) -> σmap(y[i])) for i in 1:N)
    @test_throws ArgumentError marginal_ev_system([[0.5, 0.5]])

    if N == 2
        @test_throws ArgumentError system_description(B12, m1)
    else
        sm = second_moment_system(full_second_order_marginals_system(B12))
        @test all(sm[i][1] ≈ expv((x, y) -> σmap(x[i]) * σmap(x[i+1])) &&
                  sm[i][2] ≈ expv((x, y) -> σmap(y[i]) * σmap(y[i+1])) for i in 1:N-1)
        corr = correlation_between_spins_system(sm, ev)
        pearson(a, b, c) = (a - b * c) / sqrt(abs(1 - b^2) * abs(1 - c^2))
        @test all(corr[i][1] ≈ pearson(sm[i][1], ev[i][1], ev[i+1][1]) for i in 1:N-1)
        E = energy_function(ev, sm, m1)
        @test E[1] ≈ expv((x, y) -> chain_energy(m1, x)) && E[2] ≈ expv((x, y) -> chain_energy(m1, y))

        obs = system_description(B12, m1)
        @test obs.energy == E && obs.full_expected_values == ev && obs.full_correlations == corr
        obs_t = system_description_over_time([B12, B12], m1)
        @test length(obs_t.energy_t) == 2 && obs_t.energy_t[2] == E
        @test obs_t.full_marginals_t[1] == obs.full_marginals
    end
end

@testset "time correlations = dense (N = $N)" for N in (2, 3)
    m = random_chain_model(N; rng = MersenneTwister(110 + N), β = 0.8)
    A = build_transition_tensortrain(m; update_rule = :metropolis, bond = 16)
    W = dense_operator(A, 2)
    configs = all_configs(N, 2)
    for (B0, p0) in ((normalized_boltzmann(m), boltzmann_exact(m)),
                     (TTS._product_tt(random_P0(N)), nothing))
        p0 === nothing && (p0 = tt_values(B0, 2))
        t = 5
        Bs, moments, corrs = times_correlation_evolution(A, B0, t, 16)
        @test length(Bs) == length(moments) == length(corrs) == t
        for s in 1:t
            Ws = W^s
            joint = [p0[a] * Ws[a, b] for a in eachindex(configs), b in eachindex(configs)]
            mom = [sum(joint[a, b] * σmap(configs[a][i]) * σmap(configs[b][i]) for a in axes(joint, 1), b in axes(joint, 2)) for i in 1:N]
            m0 = [sum(joint[a, b] * σmap(configs[a][i]) for a in axes(joint, 1), b in axes(joint, 2)) for i in 1:N]
            mt = [sum(joint[a, b] * σmap(configs[b][i]) for a in axes(joint, 1), b in axes(joint, 2)) for i in 1:N]
            @test moments[s] ≈ mom
            @test corrs[s] ≈ mom .- m0 .* mt                           # bug 5: ⟨σ₀⟩⟨σₜ⟩, not ⟨σₜ⟩²
        end
    end
    # bug 6: the P0 method works and equals the TT method on the product TT
    P0 = random_P0(N)
    _, mP, cP = times_correlation_evolution(A, P0, 4, 16)
    _, mT, cT = times_correlation_evolution(A, TTS._product_tt(P0), 4, 16)
    @test mP ≈ mT && cP ≈ cT
    @test_throws ArgumentError full_tcorr_system(TensorTrain([rand(1, 1, 3, 3) for _ in 1:N]))
end

@testset "cluster_size_prob = enumeration (N = $N)" for N in (2, 3, 4, 5)
    m = random_chain_model(N; rng = MersenneTwister(120 + N), β = 0.7)
    B = normalized_boltzmann(m)
    p = boltzmann_exact(m)
    configs = all_configs(N, 2)
    function sites_in_domains(x, l)
        count = 0
        for i in 1:N
            lo = i; hi = i
            while lo > 1 && x[lo-1] == x[i]; lo -= 1; end
            while hi < N && x[hi+1] == x[i]; hi += 1; end
            count += (hi - lo + 1) == l
        end
        return count
    end
    probs = [cluster_size_prob(B, l) for l in 1:N]                     # bug C: l = N included
    @test probs ≈ [sum(p[k] * sites_in_domains(x, l) / N for (k, x) in enumerate(configs)) for l in 1:N]
    @test sum(probs) ≈ 1
    @test_throws ArgumentError cluster_size_prob(B, 0)
    @test_throws ArgumentError cluster_size_prob(B, N + 1)
    @test only(portion_of_TT(B, 1, N, 2)) ≈ p[end]                       # all-up configuration
end
