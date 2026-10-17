using TensorTrainStochastics
using TensorTrains
using LinearAlgebra
using Random
using Test

include("helpers.jl")

@testset "TensorTrainStochastics.jl" begin
    @testset "core" begin
        include("core/test_models.jl")
        include("core/test_tt_utilities.jl")
    end
    @testset "lattices" begin
        include("lattices/test_lattices.jl")
    end
    @testset "distributions" begin
        include("distributions/test_distributions.jl")
    end
    @testset "transitions" begin
        include("transitions/test_transitions.jl")
    end
    @testset "evolution" begin
        include("evolution/test_evolution.jl")
    end
    @testset "observables" begin
        include("observables/test_observables.jl")
    end
end
