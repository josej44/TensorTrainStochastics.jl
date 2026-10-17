# Sequential Metropolis transition operator on an arbitrary graph (GraphModel).
# Event i (update spin i) is an MPO whose bonds carry the spins of the neighbours
# of i that are still "open" at each cut; the operator is (1/N) Σᵢ Wᵢ.

# ─────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────

function _neighbors(A::AbstractMatrix, i::Int)
    N = size(A, 1)
    return sort([j for j in 1:N if j ≠ i && A[i, j] ≠ 0.0])
end

# Channels (open neighbour spins) carried by each bond for event i.
function _channels(A::AbstractMatrix, i::Int)
    N      = size(A, 1)
    nbrs   = _neighbors(A, i)
    before = filter(j -> j < i, nbrs)
    after  = filter(j -> j > i, nbrs)
    ch     = Vector{Vector{Int}}(undef, N + 1)
    ch[1]  = Int[]
    for pos in 1:N
        curr = copy(ch[pos])
        pos ∈ before && push!(curr, pos)
        if pos == i
            filter!(c -> !(c ∈ before), curr)
            foreach(j -> push!(curr, j), after)
        end
        pos ∈ after && filter!(c -> c ≠ pos, curr)
        ch[pos + 1] = curr
    end
    return ch, before, after, nbrs
end

_bdim(ch::Vector{Int}, Q::Int) = Q ^ length(ch)

function _encode(state::Vector{Int}, Q::Int)
    isempty(state) && return 1
    foldl((acc, s) -> Q * acc + (s - 1), state; init = 0) + 1
end

function _decode(idx::Int, nchan::Int, Q::Int)
    nchan == 0 && return Int[]
    state = Vector{Int}(undef, nchan)
    r = idx - 1
    for k in nchan:-1:1
        state[k] = (r % Q) + 1
        r ÷= Q
    end
    return state
end

function _metro_rate(A::AbstractMatrix, i::Int,
                     nbr_idx::Vector{Int}, s_old::Int, s_new::Int;
                     β::Float64, Q::Int, σmap::Function)
    nbrs  = _neighbors(A, i)
    h_eff = A[i, i] + sum(A[i, nbrs[k]] * σmap(nbr_idx[k]) for k in eachindex(nbrs); init = 0.0)
    ΔE    = -(σmap(s_new) - σmap(s_old)) * h_eff * β
    return min(1.0, exp(-ΔE))
end

# ─────────────────────────────────────────────────────────────────
#  MPO cores of the event "update spin i"
# ─────────────────────────────────────────────────────────────────

function _event_cores(A::AbstractMatrix, i::Int;
                      β::Float64, Q::Int, σmap::Function)
    N = size(A, 1)
    ch, before, after, nbrs = _channels(A, i)
    cores = Vector{Array{Float64,4}}(undef, N)

    for pos in 1:N
        ch_in  = ch[pos];     ch_out = ch[pos + 1]
        d_in   = _bdim(ch_in, Q);  d_out = _bdim(ch_out, Q)
        T      = zeros(d_in, d_out, Q, Q)

        if pos == i
            for α_in in 1:d_in
                lv = _decode(α_in, length(ch_in), Q)
                for α_out in 1:d_out
                    rv = _decode(α_out, length(ch_out), Q)
                    nbr_idx = Vector{Int}(undef, length(nbrs))
                    for (k, j) in enumerate(nbrs)
                        p = findfirst(==(j), j ∈ before ? before : after)
                        nbr_idx[k] = j ∈ before ? lv[p] : rv[p]
                    end
                    for s_old in 1:Q
                        s_flip = 3 - s_old
                        r = _metro_rate(A, i, nbr_idx, s_old, s_flip; β, Q, σmap)
                        T[α_in, α_out, s_old, s_old ] += 1.0 - r
                        T[α_in, α_out, s_old, s_flip] += r
                    end
                end
            end

        elseif pos ∈ before
            for α_in in 1:d_in
                prev = _decode(α_in, length(ch_in), Q)
                for s in 1:Q
                    T[α_in, _encode(vcat(prev, [s]), Q), s, s] = 1.0
                end
            end

        elseif pos ∈ after
            chan_k = findfirst(==(pos), ch_in)
            for α_in in 1:d_in
                st = _decode(α_in, length(ch_in), Q)
                s  = st[chan_k]
                st_out = vcat(st[1:chan_k-1], st[chan_k+1:end])
                α_out  = isempty(st_out) ? 1 : _encode(st_out, Q)
                T[α_in, α_out, s, s] = 1.0
            end

        else  # pass-through
            for α in 1:d_in, s in 1:Q
                T[α, α, s, s] = 1.0
            end
        end

        cores[pos] = T
    end
    return cores
end

# ─────────────────────────────────────────────────────────────────
#  Sum of MPOs (block-diagonal in the bonds)
# ─────────────────────────────────────────────────────────────────

function _add_cores(c1::Vector{Array{Float64,4}},
                    c2::Vector{Array{Float64,4}})
    N = length(c1)
    out = Vector{Array{Float64,4}}(undef, N)
    for k in 1:N
        A1 = c1[k];  A2 = c2[k]
        d1l, d1r, Qo, Qi = size(A1)
        d2l, d2r, _,  _  = size(A2)
        if k == 1
            C = zeros(1, d1r + d2r, Qo, Qi)
            C[1, 1:d1r,     :, :] .= A1[1, :, :, :]
            C[1, d1r+1:end, :, :] .= A2[1, :, :, :]
        elseif k == N
            C = zeros(d1l + d2l, 1, Qo, Qi)
            C[1:d1l,     1, :, :] .= A1[:, 1, :, :]
            C[d1l+1:end, 1, :, :] .= A2[:, 1, :, :]
        else
            C = zeros(d1l + d2l, d1r + d2r, Qo, Qi)
            C[1:d1l,     1:d1r,     :, :] .= A1
            C[d1l+1:end, d1r+1:end, :, :] .= A2
        end
        out[k] = C
    end
    return out
end

# ─────────────────────────────────────────────────────────────────
#  Main builder: W_seq = (1/N) Σ_i W_i, compressed to `bond` (0 = no compression)
# ─────────────────────────────────────────────────────────────────

function build_transition_tensortrain(g::GraphModel;
                                      update_rule::Symbol = :metropolis,
                                      bond::Int     = 10,
                                      Q::Int        = 2,
                                      σ::Function   = s -> 2s - 3) :: TensorTrain
    update_rule == :metropolis ||
        throw(ArgumentError("GraphModel only supports update_rule = :metropolis, got :$update_rule"))
    Q == 2 || throw(ArgumentError("the graph builder only supports Q = 2, got Q = $Q"))
    A     = g.A
    β     = g.β
    σmap  = σ
    N     = size(A, 1)
    cores = _event_cores(A, 1; β, Q, σmap)
    for i in 2:N
        cores = _add_cores(cores, _event_cores(A, i; β, Q, σmap))
    end
    cores[1] ./= N          # normalize: W_seq = (1/N) Σ_i W_i
    W = TensorTrain(cores)
    bond > 0 && compress!(W; svd_trunc = TruncBond(bond))
    return W
end
