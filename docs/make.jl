using TensorTrainStochastics
using Documenter

DocMeta.setdocmeta!(TensorTrainStochastics, :DocTestSetup, :(using TensorTrainStochastics); recursive=true)

makedocs(;
    modules=[TensorTrainStochastics],
    authors="josej44 <jose.diaz@matcom.uh.cu> and contributors",
    sitename="TensorTrainStochastics.jl",
    format=Documenter.HTML(;
        canonical="https://josej44.github.io/TensorTrainStochastics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/josej44/TensorTrainStochastics.jl",
    devbranch="main",
)
