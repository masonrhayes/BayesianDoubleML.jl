using Documenter
using BayesianDoubleML
using DocStringExtensions

DocMeta.setdocmeta!(BayesianDoubleML, :DocTestSetup, :(using BayesianDoubleML); recursive=true)

makedocs(
    sitename = "BayesianDoubleML.jl",
    modules = [BayesianDoubleML],
    format = Documenter.HTML(
        mathengine = MathJax3(),
        prettyurls = false,
        canonical = "https://masonrhayes.github.io/BayesianDoubleML.jl/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "User Guide" => "user_guide.md",
        "API Reference" => "api_reference.md",
    ],
    doctest = true,
    checkdocs = :public,
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/masonrhayes/BayesianDoubleML.jl.git",
    devbranch = "master",
    push_preview = true
)
