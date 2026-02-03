mutable struct RotaryEmbedding
    inv_freq::Vector{Float32}
    cos_cached::Union{Nothing,Array{Float32,2}}
    sin_cached::Union{Nothing,Array{Float32,2}}
    seq_len_cached::Int
end

function RotaryEmbedding(dim::Int)
    inv_freq = 1.0f0 ./ (10000.0f0 .^ (Float32.(0:2:(dim-2)) ./ Float32(dim)))
    return RotaryEmbedding(inv_freq, nothing, nothing, 0)
end

function rotate_half(x::AbstractArray)
    d = size(x, 1)
    @assert iseven(d) "Rotary head_dim must be even"
    half = d ÷ 2
    x1 = view(x, 1:half, :, :)
    x2 = view(x, (half + 1):d, :, :)
    return vcat(-x2, x1)
end

function _update_cos_sin!(rot::RotaryEmbedding, seq_len::Int)
    if rot.seq_len_cached != seq_len || rot.cos_cached === nothing
        t = Float32.(0:(seq_len - 1))
        freqs = t .* reshape(rot.inv_freq, 1, :)
        emb = hcat(freqs, freqs) # (seq_len, dim)
        rot.cos_cached = transpose(cos.(emb))
        rot.sin_cached = transpose(sin.(emb))
        rot.seq_len_cached = seq_len
    end
    return rot.cos_cached::Array{Float32,2}, rot.sin_cached::Array{Float32,2}
end

function apply_rotary_pos_emb(x::AbstractArray{T,3}, cos::AbstractArray, sin::AbstractArray) where {T}
    # x: (d, seq_len, batch_heads)
    d, seq_len, _ = size(x)
    cos_view = reshape(cos[:, 1:seq_len], d, seq_len, 1)
    sin_view = reshape(sin[:, 1:seq_len], d, seq_len, 1)
    return (x .* cos_view) .+ (rotate_half(x) .* sin_view)
end

function (rot::RotaryEmbedding)(q::AbstractArray{T,3}, k::AbstractArray{T,3}) where {T}
    _, seq_len, _ = size(k)
    cos, sin = _update_cos_sin!(rot, seq_len)
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    return q_rot, k_rot
end

