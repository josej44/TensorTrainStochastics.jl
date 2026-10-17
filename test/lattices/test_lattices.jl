is_binary_symmetric(adj) = issymmetric(adj) && all(x -> x == 0 || x == 1, adj) && all(iszero, diag(adj))
degrees(adj) = vec(sum(adj; dims = 2))
bandwidth(adj) = maximum((abs(i - j) for i in axes(adj, 1), j in axes(adj, 2) if adj[i, j] != 0); init = 0)

@testset "random_3regular" begin
    for N in (4, 6, 10)
        adj = random_3regular(N; rng = MersenneTwister(N))
        @test size(adj) == (N, N)
        @test is_binary_symmetric(adj)
        @test all(==(3), degrees(adj))
    end
    @test random_3regular(8; rng = MersenneTwister(1)) == random_3regular(8; rng = MersenneTwister(1))
    for N in (2, 3, 5, 7)
        @test_throws ArgumentError random_3regular(N)
    end
end

@testset "reorder_rcm" begin
    # a path 1-2-…-6 with scrambled labels: RCM recovers bandwidth 1
    perm = [4, 1, 6, 2, 5, 3]
    path = zeros(6, 6)
    for k in 1:5
        path[perm[k], perm[k+1]] = path[perm[k+1], perm[k]] = 1.0
    end
    @test bandwidth(path) > 1
    r = reorder_rcm(path)
    @test is_binary_symmetric(r) && sum(r) == sum(path)
    @test bandwidth(r) == 1

    # relabelling preserves the graph (degree sequence, edge count)
    adj = random_3regular(12; rng = MersenneTwister(2))
    r = reorder_rcm(adj)
    @test is_binary_symmetric(r) && sum(r) == sum(adj)
    @test sort(degrees(r)) == sort(degrees(adj))
    @test bandwidth(r) <= bandwidth(adj)

    # N = 2 and N = 3, including an isolated node
    @test reorder_rcm([0.0 1.0; 1.0 0.0]) == [0.0 1.0; 1.0 0.0]
    iso = [0.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0]                 # edge 1-3, node 2 isolated
    r3 = reorder_rcm(iso)
    @test is_binary_symmetric(r3) && sum(r3) == 2 && sort(degrees(r3)) == [0, 1, 1]
    @test reorder_rcm(zeros(3, 3)) == zeros(3, 3)
end

@testset "add_gaussian_weights" begin
    adj = reorder_rcm(random_3regular(8; rng = MersenneTwister(3)))
    A = add_gaussian_weights(adj; sJ = 0.5, sh = 0.2, rng = MersenneTwister(4))
    @test issymmetric(A)
    offdiag = [(i, j) for i in 1:8, j in 1:8 if i != j]
    @test all((A[i, j] != 0) == (adj[i, j] != 0) for (i, j) in offdiag)
    @test all(!iszero, diag(A))
    @test A == add_gaussian_weights(adj; sJ = 0.5, sh = 0.2, rng = MersenneTwister(4))

    # draw order: upper-triangle couplings row by row, then fields
    rng = MersenneTwister(4)
    edges = [(i, j) for i in 1:8 for j in i+1:8 if adj[i, j] != 0]
    Jexp = [0.5 * randn(rng) for _ in edges]
    hexp = [0.2 * randn(rng) for _ in 1:8]
    @test [A[i, j] for (i, j) in edges] == Jexp && diag(A) == hexp

    # sJ = 0 leaves only the fields; N = 2 works
    @test count(!iszero, add_gaussian_weights(adj; sJ = 0.0)) == 8
    A2 = add_gaussian_weights([0.0 1.0; 1.0 0.0]; rng = MersenneTwister(5))
    @test issymmetric(A2) && A2[1, 2] != 0
end

@testset "build_instance" begin
    A = build_instance(10; sJ = 0.5, sh = 0.2, seed = 7)
    @test A == build_instance(10; sJ = 0.5, sh = 0.2, seed = 7)
    @test A != build_instance(10; sJ = 0.5, sh = 0.2, seed = 8)
    @test all(==(3), degrees(A .- Diagonal(diag(A)) .!= 0))
    @test nsites(GraphModel(A, 1.0)) == 10
    # same as running the pipeline by hand with one rng
    rng = MersenneTwister(7)
    adj = reorder_rcm(random_3regular(10; rng))
    @test A == add_gaussian_weights(adj; sJ = 0.5, sh = 0.2, rng)
    @test_throws ArgumentError build_instance(3)
end
