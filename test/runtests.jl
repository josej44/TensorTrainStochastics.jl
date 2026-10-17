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
end
