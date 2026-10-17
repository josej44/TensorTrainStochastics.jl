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
