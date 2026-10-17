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
