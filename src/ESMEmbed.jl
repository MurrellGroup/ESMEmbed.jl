module ESMEmbed

using LinearAlgebra
using Statistics

using Einops
using Flux
using NNlib
using Onion
using NPZ
using JSON
using SpecialFunctions
using HuggingFaceApi

include("safetensors.jl")

export Alphabet, RestypeTable, Alphabet_from_architecture
export ESM2Config, ESM2
export ESMFoldEmbedConfig, ESMFoldEmbed
export load_esmfold_npz!, load_esm2_npz!, load_esmfold_safetensors!
export load_ESM
export sequence_to_af2_indices

include("alphabet.jl")
include("rotary.jl")
include("attention.jl")
include("esm2.jl")
include("esmfold_embed.jl")
include("weights.jl")

end
