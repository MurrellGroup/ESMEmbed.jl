using NPZ
using JSON
using ESMEmbed
using Statistics
using NNlib

path = get(ENV, "ESMFOLD_NPZ", "esmfold_embed_ref.npz")
data = NPZ.npzread(path)
meta_path = path * ".meta.json"
meta = isfile(meta_path) ? JSON.parsefile(meta_path) : Dict{String,Any}()
config = haskey(meta, "config") ? meta["config"] : Dict{String,Any}()

num_layers = get(config, "esm_num_layers", nothing)
embed_dim = get(config, "esm_embed_dim", nothing)
attention_heads = get(config, "esm_attention_heads", nothing)
num_layers = num_layers === nothing ? nothing : num_layers
embed_dim = embed_dim === nothing ? nothing : embed_dim
attention_heads = attention_heads === nothing ? nothing : attention_heads

num_layers === nothing && error("esm_num_layers missing in metadata")
embed_dim === nothing && error("esm_embed_dim missing in metadata")

if attention_heads === nothing
    attention_heads = 20
end

c_s = get(config, "c_s", nothing)
c_s === nothing && error("c_s missing in metadata")
c_z = get(config, "c_z", 1)
c_z = c_z === nothing ? 1 : c_z

alphabet = Alphabet_from_architecture("ESM-1b")

esm = ESM2(
    num_layers,
    embed_dim,
    attention_heads;
    alphabet=alphabet,
    token_dropout=true,
)

model = ESMFoldEmbed(
    esm;
    c_s=c_s,
    c_z=c_z,
    use_esm_attn_map=false,
)

weights_path = get(ENV, "ESMFOLD_SAFETENSORS", "weights/esm.safetensors")
load_esmfold_safetensors!(model, weights_path)

sample_aa = data["sample_aa"]
ref = data["sample_s_s_0"]

aa_int = Int.(sample_aa)

# Optional intermediate comparisons
has_esm_s = haskey(data, "sample_esm_s")
has_preemb = haskey(data, "sample_s_s_0_preemb")
has_repr0 = haskey(data, "sample_repr0")
has_repr1 = haskey(data, "sample_repr1")

esm_s = nothing
if has_esm_s || has_preemb || has_repr0 || has_repr1
    mask = ones(Int, size(aa_int))
    esmaa = ESMEmbed._af2_idx_to_esm_idx(model, aa_int, mask)
    esm_s_full, _ = ESMEmbed._compute_language_model_representations(model, esmaa; need_attn=false)
    weights = NNlib.softmax(model.esm_s_combine)
    weights_view = reshape(weights, 1, 1, length(weights), 1)
    esm_s = sum(esm_s_full .* weights_view, dims=3)
    esm_s = dropdims(esm_s, dims=3)
end

if has_repr0
    ref_repr0 = data["sample_repr0"][:, 2:(end - 1), :]
    repr0 = esm_s_full[:, :, 1, :]
    max_abs = maximum(abs.(repr0 .- ref_repr0))
    mean_abs = mean(abs.(repr0 .- ref_repr0))
    println("repr0 max_abs_diff=", max_abs)
    println("repr0 mean_abs_diff=", mean_abs)
end

if has_repr1
    ref_repr1 = data["sample_repr1"][:, 2:(end - 1), :]
    repr1 = esm_s_full[:, :, 2, :]
    max_abs = maximum(abs.(repr1 .- ref_repr1))
    mean_abs = mean(abs.(repr1 .- ref_repr1))
    println("repr1 max_abs_diff=", max_abs)
    println("repr1 mean_abs_diff=", mean_abs)
end

if has_esm_s
    ref_esm_s = data["sample_esm_s"]
    max_abs = maximum(abs.(esm_s .- ref_esm_s))
    mean_abs = mean(abs.(esm_s .- ref_esm_s))
    println("esm_s max_abs_diff=", max_abs)
    println("esm_s mean_abs_diff=", mean_abs)
end

if has_preemb
    ref_preemb = data["sample_s_s_0_preemb"]
    esm_s_cf = permutedims(esm_s, (3, 2, 1))
    pred_preemb = model.esm_s_mlp(esm_s_cf)
    pred_preemb = permutedims(pred_preemb, (3, 2, 1))
    max_abs = maximum(abs.(pred_preemb .- ref_preemb))
    mean_abs = mean(abs.(pred_preemb .- ref_preemb))
    println("s_s_0_preemb max_abs_diff=", max_abs)
    println("s_s_0_preemb mean_abs_diff=", mean_abs)
end

pred = permutedims(model(aa_int), (3, 2, 1))

max_abs = maximum(abs.(pred .- ref))
mean_abs = mean(abs.(pred .- ref))

println("max_abs_diff=", max_abs)
println("mean_abs_diff=", mean_abs)
