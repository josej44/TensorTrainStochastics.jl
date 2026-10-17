isdefined(@__MODULE__, :ReferenceMC) || include(joinpath(@__DIR__, "ReferenceMC.jl"))
const RMC = ReferenceMC

# max |mc - tt| in units of the binomial standard error of a ±1 average over n samples
max_z(mc, tt, n) = maximum(abs(a - b) / (sqrt(max(1 - b^2, 1e-3)) / sqrt(n)) for (a, b) in zip(mc, tt))

@testset "bug A: Metropolis swap leaves π₁ ⊗ π₂ invariant (N = $N)" for N in (2, 3)
    m = random_chain_model(N; rng = MersenneTwister(130 + N), β = 0.4)
    β1, β2 = 0.4, 0.9
    configs = all_configs(N, 2)
    n = length(configs)
    E = [RMC.compute_total_energy(σmap.(x), m.J, m.h) for x in configs]
    π1 = boltzmann_exact(m); π2 = boltzmann_exact(with_beta(m, β2))
    π = vec(π1 * π2')                                     # π[a + n(b-1)] = π₁(a) π₂(b)

    # exact kernel of one swap attempt between chain 1 (state a) and chain 2 (state b)
    function swap_kernel(delta)
        K = zeros(n^2, n^2)
        for a in 1:n, b in 1:n
            k, ks = a + n * (b - 1), b + n * (a - 1)
            acc = min(1.0, exp(-delta(β1, β2, E[a], E[b])))
            K[k, ks] += acc
            K[k, k] += 1 - acc
        end
        return K
    end
    @test maximum(abs.(π' * swap_kernel(RMC._delta_E_swap) .- π')) < 1e-12
    original_rule(bi, bj, Ei, Ej) = (bi - bj) * (Ei - Ej)
    @test maximum(abs.(π' * swap_kernel(original_rule) .- π')) > 1e-3   # documents the bug

    # the swap functions accept with exactly that probability
    pairs = [(a, b) for a in 1:n, b in 1:n if 0.1 < min(1, exp(-RMC._delta_E_swap(β1, β2, E[a], E[b]))) < 0.9]
    a, b = first(pairs)
    acc = min(1.0, exp(-RMC._delta_E_swap(β1, β2, E[a], E[b])))
    rng = MersenneTwister(7)
    trials = 40_000
    hits1 = hits2 = hits3 = 0
    for _ in 1:trials
        chains = [σmap.(configs[a]), σmap.(configs[b])]
        hits1 += RMC.apply_metropolis_swap_with_energies!(chains, [E[a], E[b]], [β1, β2], rng)[2]
        chains = [σmap.(configs[a]), σmap.(configs[b])]
        hits2 += RMC.apply_metropolis_swap!(chains, [β1, β2], m.J, m.h, rng)[2]
        hits3 += only(RMC.compute_swap_acceptance_rate([σmap.(configs[a]), σmap.(configs[b])], [β1, β2], m.J, m.h, 1, rng))
    end
    se = sqrt(acc * (1 - acc) / trials)
    @test abs(hits1 / trials - acc) < 5se
    @test abs(hits2 / trials - acc) < 5se
    @test abs(hits3 / trials - acc) < 5se
end

@testset "basic pieces" begin
    m = random_chain_model(4; rng = MersenneTwister(140), β = 0.7)
    for x in all_configs(4, 2)
        s = σmap.(x)
        @test RMC.compute_total_energy(s, m.J, m.h) ≈ chain_energy(m, x)
        for i in 1:4
            sf = copy(s); sf[i] = -sf[i]
            @test RMC.compute_local_energy_change(s, i, m.J, m.h) ≈
                  RMC.compute_total_energy(sf, m.J, m.h) - RMC.compute_total_energy(s, m.J, m.h)
            nb = chain_neighbors(s, i)
            for s_new in (-1, 1)
                @test RMC.glauber_transition_rate(nb, s_new, i, m.β, m.J, m.h, 0.25) ≈
                      transition_rate_inertia(nb, s_new, i, m; p0 = 0.25)
            end
        end
    end
    p = RMC.MCParameters(m; betas = [0.5, 1.0], p0 = 0.1, s = 0.2, seed = 9)
    @test p.N == 4 && p.betas == [0.5, 1.0] && p.j_vector == m.J && p.h_vector == m.h
    @test p.p0 == 0.1 && p.s == 0.2 && p.seed == 9 && RMC.n_chains(p) == 2 && RMC.validate(p)
    @test RMC.MCParameters(m).betas == [m.β]

    rng = MersenneTwister(1)
    chains = [[1, 1], [-1, -1]]
    @test RMC.apply_fixed_rate_swap!(chains, 0.0, rng) == 0 && chains == [[1, 1], [-1, -1]]
    @test RMC.apply_fixed_rate_swap!(chains, 1.0, rng) == 1 && chains == [[-1, -1], [1, 1]]
    @test RMC.initialize_spins(3, [0.0, 1.0, 0.0], rng) == [-1, 1, -1]
    @test RMC.suggest_beta_schedule(0.5, 2.0, 3) ≈ [0.5, 1.0, 2.0]
    @test RMC.suggest_beta_schedule(0.5, 2.0, 4; distribution = :linear) ≈ [0.5, 1.0, 1.5, 2.0]
end

@testset "TT vs MC: single chain, $rule" for rule in (:sequential, :metropolis, :zero_temp, :parallel)
    N, T, Ns = 3, 25, 20_000
    m = random_chain_model(N; rng = MersenneTwister(150), β = 0.8)
    p = [0.2, 0.7, 0.5]
    A = build_transition_tensortrain(m; update_rule = rule, bond = 16, p0 = 0.2)
    Bt = distribution_b_tt(A, transform_init_cond_simple(p), T, 16)
    res = RMC.run_monte_carlo_general(RMC.MCParameters(m; p0 = 0.2, seed = 3);
                                      initial_probs = p, T_steps = T, update_rule = rule,
                                      N_samples = Ns, observables = [:magnetization, :energy])
    mc_mag = res[:observables][:magnetization]                       # N × (T+1)
    tt_mag = hcat([full_simple_ev_system(B) for B in Bt]...)
    @test max_z(mc_mag, tt_mag, Ns) < 5
    tt_E = [energy_function_simple(B, m) for B in Bt]
    @test maximum(abs.(res[:observables][:energy] .- tt_E) ./ max.(res[:observables][:energy_error], 1e-3)) < 5
end

@testset "TT vs MC: two replicas, $swap swap" for swap in (:fixed_rate, :metropolis)
    N, T, Ns = 3, 20, 20_000
    m1 = random_chain_model(N; rng = MersenneTwister(160), β = 0.5)
    m2 = with_beta(m1, 1.2)
    p = [0.3, 0.6, 0.8]
    A12 = parallel_transition_tensor_train(m1, m2; update_rule = :metropolis, bond = 16)
    swapfun = if swap == :fixed_rate
        ConstantSwap(0.3)
    else
        r(c) = (xy = split_replicas(c); exp((m1.β - m2.β) * (chain_energy(m1, xy[1]) - chain_energy(m1, xy[2]))))
        EnergySwap(tt_from_function(c -> min(1.0, r(c)), N, 4))      # exact Metropolis acceptance
    end
    Bt = distribution_b_tt(A12, transform_init_cond_multiple(p), T, 32, swapfun)
    res = RMC.run_monte_carlo_general(RMC.MCParameters(m1; betas = [m1.β, m2.β], s = 0.3, seed = 4);
                                      initial_probs = p, T_steps = T, update_rule = :metropolis,
                                      swap_criterion = swap, N_samples = Ns, observables = [:magnetization])
    mc_mag = res[:observables][:magnetization]                       # 2 × N × (T+1)
    ev = [marginal_ev_system(full_marginals_system(B)) for B in Bt]
    for c in 1:2
        tt_mag = [ev[t][i][c] for i in 1:N, t in 1:T+1]
        @test max_z(mc_mag[c, :, :], tt_mag, Ns) < 5
    end
end

@testset "TT vs MC: graph Metropolis" begin
    N, T, Ns = 4, 30, 20_000
    g = GraphModel(build_instance(N; seed = 3), 0.8)
    W = build_transition_tensortrain(g; bond = 32)
    Bt = distribution_b_tt(W, [[0.5, 0.5] for _ in 1:N], T, 32)
    tt_mag = hcat([full_simple_ev_system(B) for B in Bt]...)'          # (T+1) × N
    mc_mag = RMC.montecarlo_magnetizations_single(g.A, T; n_chains = Ns, β = g.β, seed = 5)
    @test max_z(mc_mag, tt_mag, Ns) < 5
    # a graph with an isolated node (bug H) runs
    iso = [0.3 0.0 1.0; 0.0 -0.2 0.0; 1.0 0.0 0.1]
    @test size(RMC.montecarlo_magnetizations_single(iso, 3; n_chains = 10)) == (4, 3)
end

@testset "sample_boltzmann_ising (bug D) and cluster_size_prob_mc" begin
    m = ChainModel([0.5, -0.3, 0.8], [0.9, -0.6, 0.4, 0.7], 0.6)
    samples = RMC.sample_boltzmann_ising(RMC.MCParameters(m); n_samples = 40_000)
    @test size(samples) == (4, 40_000) && all(x -> x == 1 || x == -1, samples)
    B = boltzmann_tt(m); normalize!(B)
    exact_mag = full_simple_ev_system(B)
    @test maximum(abs.(vec(mean(samples; dims = 2)) .- exact_mag)) < 0.03
    for l in 1:4
        @test abs(RMC.cluster_size_prob_mc(samples, l) - cluster_size_prob(B, l)) < 0.03
    end
    # exact check of cluster_size_prob_mc on fixed configurations
    @test RMC.cluster_size_prob_mc([1 1; 1 -1; -1 -1; 1 -1], 2) == (2 + 0) / 8   # [1,1,-1,1]: sites 1,2
end
