using Documenter
using BayesianDoubleML
using DocStringExtensions
using PlutoStaticHTML
using SHA

# Build Pluto notebooks before generating documentation
println("Building Pluto notebooks...")

# Define notebook paths and their destination names
notebooks = [
    ("MCMC Example", joinpath(@__DIR__, "..", "examples", "MCMC", "mcmc_example.jl")),
    ("VI Example", joinpath(@__DIR__, "..", "examples", "VI", "vi_example.jl")),
]

# Output directory for notebooks in documentation
notebooks_output_dir = joinpath(@__DIR__, "src", "examples")
mkpath(notebooks_output_dir)

"""
    files_are_equal(path1, path2) -> Bool

Compare two files by their SHA256 checksum to determine if they have the same content.
"""
function files_are_equal(path1::String, path2::String)
    isfile(path1) && isfile(path2) || return false
    return open(path1, "r") do io1
        open(path2, "r") do io2
            return sha256(io1) == sha256(io2)
        end
    end
end

# Build each notebook and copy to docs/src/examples/
for (title, notebook_path) in notebooks
    if isfile(notebook_path)
        println("  Building: $title")

        # Get the directory containing the notebook
        notebook_dir = dirname(notebook_path)

        # Build options - output goes next to the notebook
        bopts = PlutoStaticHTML.BuildOptions(
            notebook_dir;
            output_format = PlutoStaticHTML.documenter_output,
            add_documenter_css = true,
            previous_dir = notebooks_output_dir
        )

        # Build the notebook
        PlutoStaticHTML.build_notebooks(bopts, [notebook_path])

        # The generated file has the same name but with .md extension
        notebook_name = basename(notebook_path)
        generated_md = replace(notebook_name, ".jl" => ".md")
        generated_path = joinpath(notebook_dir, generated_md)

        # Copy to docs/src/examples/ only if content changed
        dest_path = joinpath(notebooks_output_dir, generated_md)
        if files_are_equal(generated_path, dest_path)
            println("    ✓ Up to date: $dest_path")
        else
            cp(generated_path, dest_path; force = true)
            println("    ✓ Updated: $dest_path")
        end
    else
        @warn "Notebook not found: $notebook_path"
    end
end

println("✓ Pluto notebooks built successfully")

DocMeta.setdocmeta!(BayesianDoubleML, :DocTestSetup, :(using BayesianDoubleML); recursive = true)

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
        "Examples" => [
            "MCMC Example" => "examples/mcmc_example.md",
            "VI Example" => "examples/vi_example.md",
        ],
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
