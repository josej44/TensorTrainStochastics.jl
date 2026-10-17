# Monte Carlo tools that lived only in the notebooks (basic_plots, multiple_steps_cluster).

"""
    mc_step_single!(σ::Vector{Int}, A::Matrix{Float64}; β=1.0, rng=Random.GLOBAL_RNG)

One sequential Metropolis step on a graph model `A` (as in `GraphModel`); `σ` holds
indices 1/2 (spin -1/+1). From basic_plots.ipynb.
"""
function mc_step_single!(σ::Vector{Int}, A::Matrix{Float64};
                         β::Float64 = 1.0, rng = Random.GLOBAL_RNG)
    N = length(σ)
    i = rand(rng, 1:N)                         # random spin
    nbrs  = [j for j in 1:N if j ≠ i && A[i, j] ≠ 0.0]
    # init = 0.0: same fix as bug H (a site with no neighbours)
    h_eff = A[i, i] + sum(A[i, j] * (2σ[j] - 3) for j in nbrs; init = 0.0)

    s_new = 3 - σ[i]
    ΔE    = -(2s_new - 3 - (2σ[i] - 3)) * h_eff * β
    rand(rng) < min(1.0, exp(-ΔE)) && (σ[i] = s_new)
end

"""
    montecarlo_magnetizations_single(A, t; n_chains=10_000, β=1.0, P0_mode=:uniform, seed=1)

Average magnetization per site over `n_chains` independent runs of `t` sequential
Metropolis steps on the graph `A`. Returns a `(t+1) × N` matrix. `P0_mode` is
`:uniform`, `:up` or `:down`. From basic_plots.ipynb.
"""
function montecarlo_magnetizations_single(A::Matrix{Float64}, t::Int;
                                          n_chains::Int  = 10_000,
                                          β::Float64     = 1.0,
                                          P0_mode::Symbol = :uniform,
                                          seed::Int       = 1)
    N   = size(A, 1)
    rng = MersenneTwister(seed)
    acc = zeros(t + 1, N)

    for _ in 1:n_chains
        σ = if P0_mode == :uniform
            rand(rng, 1:2, N)
        elseif P0_mode == :up
            fill(2, N)
        elseif P0_mode == :down
            fill(1, N)
        else
            rand(rng, 1:2, N)
        end

        acc[1, :] .+= (2 .* σ .- 3)

        for step in 1:t
            mc_step_single!(σ, A; β, rng)      # one spin per step
            acc[step+1, :] .+= (2 .* σ .- 3)
        end
    end

    return acc ./ n_chains
end

"""
    sample_boltzmann_ising(params; n_samples, n_burnin=10_000)

Boltzmann samples of the chain (`params.betas[1]`) by single-flip Metropolis with a fixed
seed, one sample every N attempts after `n_burnin` attempts. Returns an `N × n_samples`
matrix of ±1. From multiple_steps_cluster.ipynb.
"""
function sample_boltzmann_ising(params; n_samples::Int, n_burnin::Int = 10_000)
    N        = params.N
    beta     = params.betas[1]
    J        = params.j_vector   # length N-1
    h        = params.h_vector   # length N

    rng_local = Xoshiro(42)

    # random initial ±1 configuration
    config = rand(rng_local, [-1, 1], N)

    # β·ΔE of flipping spin i (bug D: the field term was missing the factor β)
    function delta_energy(config, i)
        dE = 2.0 * beta * config[i] * h[i]
        if i > 1;   dE += 2.0 * beta * J[i-1] * config[i] * config[i-1]; end
        if i < N;   dE += 2.0 * beta * J[i]   * config[i] * config[i+1]; end
        return dE
    end

    # burn-in
    for _ in 1:n_burnin
        i = rand(rng_local, 1:N)
        dE = delta_energy(config, i)
        if dE < 0 || rand(rng_local) < exp(-dE)
            config[i] = -config[i]
        end
    end

    # sampling: keep one configuration every N attempts (one sweep)
    samples = Matrix{Int}(undef, N, n_samples)
    for s in 1:n_samples
        for _ in 1:N
            i = rand(rng_local, 1:N)
            dE = delta_energy(config, i)
            if dE < 0 || rand(rng_local) < exp(-dE)
                config[i] = -config[i]
            end
        end
        samples[:, s] = config
    end
    return samples                          # N × n_samples, values ±1
end

"""
    cluster_size_prob_mc(samples, l)

Fraction of sites (over all samples) that belong to a maximal domain of length `l`.
From multiple_steps_cluster.ipynb.
"""
function cluster_size_prob_mc(samples::Matrix{Int}, l::Int)
    N, S = size(samples)
    count = 0
    for s in 1:S
        config = view(samples, :, s)
        for i in 1:N
            # maximal block containing spin i
            left  = i
            right = i
            while left  > 1 && config[left  - 1] == config[i]; left  -= 1; end
            while right < N && config[right + 1] == config[i]; right += 1; end
            if (right - left + 1) == l
                count += 1
            end
        end
    end
    return count / (N * S)
end
