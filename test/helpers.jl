# Brute-force helpers shared by the test files (exact enumeration, small N only).

# All configurations of N sites with physical dimension Q, as vectors of indices 1:Q.
all_configs(N, Q) = vec([collect(Tuple(c)) for c in CartesianIndices(ntuple(_ -> Q, N))])

# Value of a tensor train at `idx`, where idx[i] is the index tuple of the
# physical indices of site i (an Int for 3-index cores).
tt_value(A, idx) = float(only(prod(A[i][:, :, idx[i]...] for i in eachindex(A))) / A.z)

# Dense matrix W[x, x'] of a 4-index (transition) tensor train over all configurations.
function dense_operator(A, Q)
    configs = all_configs(length(A), Q)
    return [tt_value(A, [(x[i], y[i]) for i in eachindex(x)]) for x in configs, y in configs]
end

# Maximum pointwise error |B(x) X(x) - 1| over all configurations of a 3-index TT.
function max_inverse_error(B, X)
    Q = size(B[1], 3)
    return maximum(abs(tt_value(B, x) * tt_value(X, x) - 1) for x in all_configs(length(B), Q))
end

# Run `f` and return everything it wrote to stdout and stderr.
function captured_output(f)
    mktemp() do path, io
        redirect_stdout(io) do
            redirect_stderr(io) do
                f()
            end
        end
        flush(io)
        return read(path, String)
    end
end

# ---------------------------------------------------------------------------
# Brute-force physics references
# ---------------------------------------------------------------------------

const σmap = x -> 2x - 3

# Energy of a chain at index configuration x (x[i] ∈ 1:2).
function chain_energy(m::ChainModel, x)
    s = σmap.(x)
    return -sum(m.J[i] * s[i] * s[i+1] for i in eachindex(m.J)) - sum(m.h .* s)
end

# Energy of a graph model at index configuration x.
function graph_energy(g::GraphModel, x)
    s = σmap.(x); N = length(s)
    return -sum(g.A[i, j] * s[i] * s[j] for i in 1:N for j in i+1:N) - sum(g.A[i, i] * s[i] for i in 1:N)
end

# Exact normalized Boltzmann distribution over all_configs(N, 2).
function boltzmann_exact(m::ChainModel)
    w = [exp(-m.β * chain_energy(m, x)) for x in all_configs(nsites(m), 2)]
    return w ./ sum(w)
end

# From new_reviser.ipynb / boltzmann_distribution_tt.jl, adapted to models:
# x[i] = (s_i, s_i') holds the index of replica 1 and replica 2 at site i.
function energy_difference(m1::ChainModel, m2::ChainModel, x)
    σ = x -> 2x - 3
    e_diff = m1.h[1] * (σ(x[1][1]) - σ(x[1][2]))
    N = nsites(m1)
    for i in 1:N-1
        s_i, s_i_prima = x[i]
        s_j, s_j_prima = x[i+1]
        e_diff += m1.J[i] * (σ(s_i) * σ(s_j) - σ(s_i_prima) * σ(s_j_prima)) + m1.h[i+1] * (σ(s_j) - σ(s_j_prima))
    end
    e_diff *= m2.β - m1.β
    return exp(e_diff)
end

function energy_difference_n(m1::ChainModel, m2::ChainModel, x, n)
    σ = x -> 2x - 3
    e_diff = m1.h[1] * (σ(x[1][1]) - σ(x[1][2]))
    N = nsites(m1)
    for i in 1:N-1
        s_i, s_i_prima = x[i]
        s_j, s_j_prima = x[i+1]
        e_diff += m1.J[i] * (σ(s_i) * σ(s_j) - σ(s_i_prima) * σ(s_j_prima)) + m1.h[i+1] * (σ(s_j) - σ(s_j_prima))
    end
    e_diff *= n * (m2.β - m1.β)
    return exp(e_diff)
end

# Combined replica index of (x, y) for Q = 2: x + 2(y - 1).
combined_index(x, y) = x .+ 2 .* (y .- 1)

# ---------------------------------------------------------------------------
# Brute-force transition operators (rows = configuration at t, cols = at t+1)
# ---------------------------------------------------------------------------

# Neighbourhood layout used by the chain rates (physical spin values).
function chain_neighbors(s, i)
    N = length(s)
    i == 1 && return [s[1], s[2]]
    i == N && return [s[N-1], s[N]]
    return [s[i-1], s[i], s[i+1]]
end

function dense_sequential(rate, m::ChainModel)
    N = nsites(m); configs = all_configs(N, 2)
    W = zeros(length(configs), length(configs))
    for (a, x) in enumerate(configs), (b, y) in enumerate(configs), i in 1:N
        all(x[k] == y[k] for k in 1:N if k != i) || continue
        W[a, b] += rate(chain_neighbors(σmap.(x), i), σmap(y[i]), i, m) / N
    end
    return W
end

function dense_parallel(rate, m::ChainModel)
    N = nsites(m); configs = all_configs(N, 2)
    return [prod(rate(chain_neighbors(σmap.(x), i), σmap(y[i]), i, m) for i in 1:N) for x in configs, y in configs]
end

function dense_graph_metropolis(g::GraphModel)
    N = nsites(g); configs = all_configs(N, 2)
    W = zeros(length(configs), length(configs))
    for (a, x) in enumerate(configs), (b, y) in enumerate(configs), i in 1:N
        all(x[k] == y[k] for k in 1:N if k != i) || continue
        xf = copy(x); xf[i] = 3 - x[i]
        acc = min(1.0, exp(-g.β * (graph_energy(g, xf) - graph_energy(g, x))))
        W[a, b] += (y[i] == x[i] ? 1 - acc : acc) / N
    end
    return W
end

function boltzmann_exact(g::GraphModel)
    w = [exp(-g.β * graph_energy(g, x)) for x in all_configs(nsites(g), 2)]
    return w ./ sum(w)
end

# Split a replica configuration (combined indices 1:4 per site) into (x, y).
split_replicas(c) = ([(k - 1) % 2 + 1 for k in c], [(k - 1) ÷ 2 + 1 for k in c])

# Exact tensor train of an arbitrary function f(x) on all_configs(N, q), built as a sum
# of bond-1 indicator TTs (small N only), then compressed without truncation.
function tt_from_function(f, N, q)
    configs = all_configs(N, q)
    terms = map(configs) do x
        TensorTrain([reshape(Float64.(1:q .== x[i]) .* (i == 1 ? f(x) : 1.0), 1, 1, q) for i in 1:N])
    end
    T = reduce(+, terms)
    compress!(T; svd_trunc = TruncThresh(0.0))
    return T
end

# Values of a 3-index TT over all configurations (same order as dense_operator).
tt_values(B, q) = [tt_value(B, x) for x in all_configs(length(B), q)]
