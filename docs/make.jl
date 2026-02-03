using ESMEmbed
using Documenter

DocMeta.setdocmeta!(ESMEmbed, :DocTestSetup, :(using ESMEmbed); recursive=true)

makedocs(;
    modules=[ESMEmbed],
    authors="Ben Murrell",
    sitename="ESMEmbed.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/ESMEmbed.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ESMEmbed.jl",
    devbranch="main",
)
