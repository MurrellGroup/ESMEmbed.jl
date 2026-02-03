@concrete struct ESMMultiheadAttention <: Onion.Layer
    q_proj
    k_proj
    v_proj
    out_proj
    num_heads::Int
    head_dim::Int
    scaling::Float32
    dropout::Float32
    rot_emb
end

@layer ESMMultiheadAttention

function ESMMultiheadAttention(
    embed_dim::Int,
    num_heads::Int;
    dropout::Real = 0.0,
    bias::Bool = true,
    use_rotary_embeddings::Bool = false,
)
    head_dim = embed_dim ÷ num_heads
    @assert head_dim * num_heads == embed_dim "embed_dim must be divisible by num_heads"

    q_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    k_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    v_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)
    out_proj = Onion.Linear(embed_dim => embed_dim; bias=bias)

    rot_emb = use_rotary_embeddings ? RotaryEmbedding(head_dim) : nothing

    return ESMMultiheadAttention(
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        num_heads,
        head_dim,
        Float32(head_dim)^-0.5f0,
        Float32(dropout),
        rot_emb,
    )
end

function _reshape_heads(x::AbstractArray, head_dim::Int, num_heads::Int)
    # x: (C, T, B) -> (D, T, B*H)
    d, t, b = head_dim, size(x, 2), size(x, 3)
    x = reshape(x, d, num_heads, t, b)
    x = permutedims(x, (1, 3, 2, 4))
    return reshape(x, d, t, b * num_heads)
end

function _unshape_heads(x::AbstractArray, head_dim::Int, num_heads::Int, batch::Int)
    # x: (D, T, B*H) -> (C, T, B)
    d, t, _ = size(x)
    x = reshape(x, d, t, num_heads, batch)
    x = permutedims(x, (1, 3, 2, 4))
    return reshape(x, d * num_heads, t, batch)
end

function _apply_key_padding_mask!(attn_weights::AbstractArray, key_padding_mask)
    # attn_weights: (T, S, B*H)
    if key_padding_mask === nothing
        return attn_weights
    end
    # key_padding_mask: (B, S) with true for padding
    s, b = size(key_padding_mask, 2), size(key_padding_mask, 1)
    mask = permutedims(key_padding_mask, (2, 1)) # (S, B)
    mask = repeat(mask, 1, size(attn_weights, 3) ÷ b) # (S, B*H)
    bias = ifelse.(mask, -Inf, 0.0)
    bias = convert.(eltype(attn_weights), bias)
    attn_weights .+= reshape(bias, 1, s, :) 
    return attn_weights
end

function (m::ESMMultiheadAttention)(
    query::AbstractArray,
    key::AbstractArray = query,
    value::AbstractArray = key;
    key_padding_mask = nothing,
    need_head_weights::Bool = false,
)
    # Inputs are (C, T, B)
    q = m.q_proj(query)
    k = m.k_proj(key)
    v = m.v_proj(value)

    q = q .* m.scaling

    q = _reshape_heads(q, m.head_dim, m.num_heads)
    k = _reshape_heads(k, m.head_dim, m.num_heads)
    v = _reshape_heads(v, m.head_dim, m.num_heads)

    if m.rot_emb !== nothing
        q, k = (m.rot_emb)(q, k)
    end

    # attn_weights: (T, S, B*H)
    attn_weights = NNlib.batched_mul(permutedims(q, (2, 1, 3)), k)
    _apply_key_padding_mask!(attn_weights, key_padding_mask)

    attn_weights_float = NNlib.softmax(Float32.(attn_weights); dims=2)
    attn_probs = attn_weights_float

    attn = NNlib.batched_mul(attn_probs, permutedims(v, (2, 1, 3)))
    attn = permutedims(attn, (2, 1, 3))
    attn = _unshape_heads(attn, m.head_dim, m.num_heads, size(query, 3))

    attn = m.out_proj(attn)

    attn_weights_out = nothing
    if need_head_weights
        t, s = size(attn_weights_float, 1), size(attn_weights_float, 2)
        b = size(query, 3)
        attn_weights_out = reshape(attn_weights_float, t, s, m.num_heads, b)
        attn_weights_out = permutedims(attn_weights_out, (4, 3, 1, 2)) # (B, H, T, S)
    end

    return attn, attn_weights_out
end
