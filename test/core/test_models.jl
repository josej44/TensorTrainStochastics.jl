@testset "ChainModel" begin
    m = ChainModel([1.0, -0.5], [0.1, 0.2, 0.3], 2.0)
    @test nsites(m) == 3
    @test m.J == [1.0, -0.5] && m.h == [0.1, 0.2, 0.3] && m.β == 2.0
    @test local_fields(m) == [0.1, 0.2, 0.3]

    # conversion from integer vectors / reals
    mi = ChainModel([1, 1], [0, 0, 0], 1)
    @test mi.J isa Vector{Float64} && mi.β === 1.0

    # inputs are copied
    J = [1.0, 2.0]
    mc = ChainModel(J, zeros(3), 1.0)
    J[1] = 99.0
    @test mc.J == [1.0, 2.0]

    # smallest chain and negative β are allowed (swap ratios use β₂ - β₁)
    @test nsites(ChainModel([0.5], [0.0, 0.0], 1.0)) == 2
    @test ChainModel([0.5], [0.0, 0.0], -1.5).β == -1.5

    @test_throws ArgumentError ChainModel(Float64[], [0.0], 1.0)        # N = 1
    @test_throws ArgumentError ChainModel([1.0], [0.0, 0.0, 0.0], 1.0)  # length(J) != N-1
    @test_throws ArgumentError ChainModel([1.0, 1.0], [0.0, 0.0], 1.0)
    @test_throws ArgumentError ChainModel([1.0], [0.0, 0.0], NaN)
end

@testset "GraphModel" begin
    A = [0.1 1.0 0.0; 1.0 -0.2 0.5; 0.0 0.5 0.3]
    g = GraphModel(A, 0.7)
    @test nsites(g) == 3
    @test local_fields(g) == [0.1, -0.2, 0.3]
    A[1, 2] = 42.0
    @test g.A[1, 2] == 1.0                                               # copied

    @test_throws ArgumentError GraphModel([1.0 2.0; 3.0 4.0], 1.0)      # not symmetric
    @test_throws ArgumentError GraphModel(zeros(2, 3), 1.0)             # not square
    @test_throws ArgumentError GraphModel(fill(1.0, 1, 1), 1.0)         # N = 1
    @test_throws ArgumentError GraphModel(zeros(2, 2), NaN)

    # GraphModel(::ChainModel) is the tridiagonal equivalent
    m = ChainModel([1.0, -2.0, 3.0], [0.1, 0.2, 0.3, 0.4], 1.5)
    gm = GraphModel(m)
    @test gm.A == [0.1 1.0 0.0 0.0; 1.0 0.2 -2.0 0.0; 0.0 -2.0 0.3 3.0; 0.0 0.0 3.0 0.4]
    @test gm.β == m.β
    @test local_fields(gm) == local_fields(m)
end

@testset "with_beta" begin
    m = ChainModel([1.0, 2.0], [0.0, 0.5, 1.0], 1.0)
    m2 = with_beta(m, 3.0)
    @test m2.β == 3.0 && m2.J == m.J && m2.h == m.h
    @test m.β == 1.0
    g2 = with_beta(GraphModel(m), -0.5)
    @test g2.β == -0.5 && g2.A == GraphModel(m).A
end

@testset "random_chain_model" begin
    m = random_chain_model(6; rng = MersenneTwister(3))
    @test m isa ChainModel && nsites(m) == 6 && length(m.J) == 5
    @test all(-1 .<= m.J .< 1) && all(-1 .<= m.h .< 1) && 0 <= m.β < 1

    # same draw order as the original random_params: β, then J, then h
    rng = MersenneTwister(3)
    β = rand(rng); J = -1 .+ 2 .* rand(rng, 5); h = -1 .+ 2 .* rand(rng, 6)
    @test (m.β, m.J, m.h) == (β, J, h)

    m2 = random_chain_model(4, 0.0, 0.5; rng = MersenneTwister(1), β = 2.0)
    @test m2.β == 2.0 && all(0 .<= m2.J .< 0.5) && all(0 .<= m2.h .< 0.5)
    @test_throws ArgumentError random_chain_model(1)
end

@testset "no export clashes with TensorTrains" begin
    ours = setdiff(names(TensorTrainStochastics), [:TensorTrainStochastics])
    @test isempty(intersect(ours, names(TensorTrains)))
end
