function _npz_state(path::AbstractString)
    data = NPZ.npzread(path)
    meta_path = path * ".names.json"
    if isfile(meta_path)
        meta = JSON.parsefile(meta_path)
        names = meta["names"]
        state = Dict{String,Any}()
        for (i, name) in enumerate(names)
            key = "arr_$(i)"
            state[name] = data[key]
        end
        return state
    end
    return Dict{String,Any}(data)
end

function _assign!(dest, src)
    dest .= src
    return dest
end

function _assign_transpose!(dest, src)
    dest .= permutedims(src, (2, 1))
    return dest
end

function load_esm2_npz!(model::ESM2, path::AbstractString)
    state = _npz_state(path)

    _assign_transpose!(model.embed_tokens.weight, state["esm.embed_tokens.weight"])

    for i in 0:(model.num_layers - 1)
        layer = model.layers[i + 1]
        prefix = "esm.layers.$i"

        _assign!(layer.self_attn.q_proj.weight, state["$prefix.self_attn.q_proj.weight"])
        _assign!(layer.self_attn.k_proj.weight, state["$prefix.self_attn.k_proj.weight"])
        _assign!(layer.self_attn.v_proj.weight, state["$prefix.self_attn.v_proj.weight"])
        _assign!(layer.self_attn.out_proj.weight, state["$prefix.self_attn.out_proj.weight"])

        _assign!(layer.self_attn.q_proj.bias, state["$prefix.self_attn.q_proj.bias"])
        _assign!(layer.self_attn.k_proj.bias, state["$prefix.self_attn.k_proj.bias"])
        _assign!(layer.self_attn.v_proj.bias, state["$prefix.self_attn.v_proj.bias"])
        _assign!(layer.self_attn.out_proj.bias, state["$prefix.self_attn.out_proj.bias"])

        _assign!(layer.self_attn_layer_norm.w, state["$prefix.self_attn_layer_norm.weight"])
        _assign!(layer.self_attn_layer_norm.b, state["$prefix.self_attn_layer_norm.bias"])

        _assign!(layer.fc1.weight, state["$prefix.fc1.weight"])
        _assign!(layer.fc1.bias, state["$prefix.fc1.bias"])
        _assign!(layer.fc2.weight, state["$prefix.fc2.weight"])
        _assign!(layer.fc2.bias, state["$prefix.fc2.bias"])

        _assign!(layer.final_layer_norm.w, state["$prefix.final_layer_norm.weight"])
        _assign!(layer.final_layer_norm.b, state["$prefix.final_layer_norm.bias"])
    end

    _assign!(model.emb_layer_norm_after.w, state["esm.emb_layer_norm_after.weight"])
    _assign!(model.emb_layer_norm_after.b, state["esm.emb_layer_norm_after.bias"])

    if haskey(state, "esm.lm_head.dense.weight")
        _assign!(model.lm_head.dense.weight, state["esm.lm_head.dense.weight"])
        _assign!(model.lm_head.dense.bias, state["esm.lm_head.dense.bias"])
        _assign!(model.lm_head.layer_norm.w, state["esm.lm_head.layer_norm.weight"])
        _assign!(model.lm_head.layer_norm.b, state["esm.lm_head.layer_norm.bias"])
        _assign!(model.lm_head.bias, state["esm.lm_head.bias"])
        # tie weights
        model.lm_head.weight .= model.embed_tokens.weight
    end

    return model
end

function load_esmfold_safetensors!(model::ESMFoldEmbed, reader::SafeTensors.Reader)
    SafeTensors.read_into!(reader, "af2_to_esm", model.af2_to_esm)
    SafeTensors.read_into!(reader, "esm_s_combine", model.esm_s_combine)

    SafeTensors.read_into!(reader, "esm_s_mlp.0.weight", model.esm_s_mlp.norm.w)
    SafeTensors.read_into!(reader, "esm_s_mlp.0.bias", model.esm_s_mlp.norm.b)
    SafeTensors.read_into!(reader, "esm_s_mlp.1.weight", model.esm_s_mlp.fc1.weight)
    SafeTensors.read_into!(reader, "esm_s_mlp.1.bias", model.esm_s_mlp.fc1.bias)
    SafeTensors.read_into!(reader, "esm_s_mlp.3.weight", model.esm_s_mlp.fc2.weight)
    SafeTensors.read_into!(reader, "esm_s_mlp.3.bias", model.esm_s_mlp.fc2.bias)

    if model.esm_z_mlp !== nothing
        SafeTensors.read_into!(reader, "esm_z_mlp.0.weight", model.esm_z_mlp.norm.w)
        SafeTensors.read_into!(reader, "esm_z_mlp.0.bias", model.esm_z_mlp.norm.b)
        SafeTensors.read_into!(reader, "esm_z_mlp.1.weight", model.esm_z_mlp.fc1.weight)
        SafeTensors.read_into!(reader, "esm_z_mlp.1.bias", model.esm_z_mlp.fc1.bias)
        SafeTensors.read_into!(reader, "esm_z_mlp.3.weight", model.esm_z_mlp.fc2.weight)
        SafeTensors.read_into!(reader, "esm_z_mlp.3.bias", model.esm_z_mlp.fc2.bias)
    end

    # embedding.weight in checkpoint is (n_tokens, c_s); Flux expects (c_s, n_tokens)
    permutedims!(
        model.embedding.weight,
        SafeTensors.read_tensor(reader, "embedding.weight"),
        (2, 1),
    )

    # ESM2 weights
    # word_embeddings in checkpoint is (vocab, dim); Flux expects (dim, vocab)
    permutedims!(
        model.esm.embed_tokens.weight,
        SafeTensors.read_tensor(reader, "esm.embeddings.word_embeddings.weight"),
        (2, 1),
    )
    SafeTensors.read_into!(reader, "esm.encoder.emb_layer_norm_after.weight", model.esm.emb_layer_norm_after.w)
    SafeTensors.read_into!(reader, "esm.encoder.emb_layer_norm_after.bias", model.esm.emb_layer_norm_after.b)

    for i in 0:(model.esm.num_layers - 1)
        layer = model.esm.layers[i + 1]
        prefix = "esm.encoder.layer.$i"

        SafeTensors.read_into!(reader, "$prefix.attention.self.query.weight", layer.self_attn.q_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.query.bias", layer.self_attn.q_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.self.key.weight", layer.self_attn.k_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.key.bias", layer.self_attn.k_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.self.value.weight", layer.self_attn.v_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.self.value.bias", layer.self_attn.v_proj.bias)
        SafeTensors.read_into!(reader, "$prefix.attention.output.dense.weight", layer.self_attn.out_proj.weight)
        SafeTensors.read_into!(reader, "$prefix.attention.output.dense.bias", layer.self_attn.out_proj.bias)

        SafeTensors.read_into!(reader, "$prefix.attention.LayerNorm.weight", layer.self_attn_layer_norm.w)
        SafeTensors.read_into!(reader, "$prefix.attention.LayerNorm.bias", layer.self_attn_layer_norm.b)

        SafeTensors.read_into!(reader, "$prefix.intermediate.dense.weight", layer.fc1.weight)
        SafeTensors.read_into!(reader, "$prefix.intermediate.dense.bias", layer.fc1.bias)
        SafeTensors.read_into!(reader, "$prefix.output.dense.weight", layer.fc2.weight)
        SafeTensors.read_into!(reader, "$prefix.output.dense.bias", layer.fc2.bias)

        SafeTensors.read_into!(reader, "$prefix.LayerNorm.weight", layer.final_layer_norm.w)
        SafeTensors.read_into!(reader, "$prefix.LayerNorm.bias", layer.final_layer_norm.b)
    end

    return model
end

function load_esmfold_safetensors!(model::ESMFoldEmbed, path::AbstractString)
    reader = SafeTensors.Reader(path)
    return load_esmfold_safetensors!(model, reader)
end

function _infer_esmfold_config(reader::SafeTensors.Reader)
    header_keys = collect(keys(reader.header))

    layer_ids = Int[]
    for key in header_keys
        startswith(key, "esm.encoder.layer.") || continue
        parts = split(key, '.')
        length(parts) < 4 && continue
        idx = tryparse(Int, parts[4])
        idx === nothing || push!(layer_ids, idx)
    end
    isempty(layer_ids) && error("Unable to infer num_layers from safetensors header.")
    num_layers = maximum(layer_ids) + 1

    embed_entry = reader.header["esm.embeddings.word_embeddings.weight"]
    embed_dim = embed_entry.shape[2]

    inv_entry = reader.header["esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"]
    head_dim = inv_entry.shape[1] * 2
    attention_heads = embed_dim ÷ head_dim

    cs_entry = reader.header["esm_s_mlp.1.weight"]
    c_s = cs_entry.shape[1]

    c_z = 1
    if haskey(reader.header, "esm_z_mlp.1.weight")
        cz_entry = reader.header["esm_z_mlp.1.weight"]
        c_z = cz_entry.shape[1]
    end

    return num_layers, embed_dim, attention_heads, c_s, c_z
end

function load_ESM(;
    repo_id::AbstractString = "facebook/esmfold_v1",
    filename::AbstractString = "model.safetensors",
    revision::AbstractString = "ba837a3",
    cache::Bool = true,
    local_files_only::Bool = false,
    use_esm_attn_map::Bool = false,
)
    path = hf_hub_download(
        repo_id,
        filename;
        revision = revision,
        cache = cache,
        local_files_only = local_files_only,
    )

    reader = SafeTensors.Reader(path)
    num_layers, embed_dim, attention_heads, c_s, c_z = _infer_esmfold_config(reader)

    use_esm_attn_map && !haskey(reader.header, "esm_z_mlp.1.weight") &&
        error("use_esm_attn_map=true but esm_z_mlp weights are not present in the checkpoint.")

    alphabet = Alphabet_from_architecture("ESM-1b")
    esm = ESM2(
        num_layers,
        embed_dim,
        attention_heads;
        alphabet = alphabet,
        token_dropout = true,
    )
    model = ESMFoldEmbed(
        esm;
        c_s = c_s,
        c_z = c_z,
        use_esm_attn_map = use_esm_attn_map,
    )
    load_esmfold_safetensors!(model, reader)
    return model
end
