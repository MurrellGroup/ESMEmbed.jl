struct RestypeTable
    restypes::Vector{String}
    restypes_with_x::Vector{String}
    restype_order::Dict{String,Int}
    restype_order_with_x::Dict{String,Int}
end

const RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

const RESTYPES_WITH_X = vcat(RESTYPES, ["X"])

const RESTYPE_ORDER = Dict{String,Int}(restype => i - 1 for (i, restype) in enumerate(RESTYPES))
const RESTYPE_ORDER_WITH_X = Dict{String,Int}(restype => i - 1 for (i, restype) in enumerate(RESTYPES_WITH_X))

const RESTYPE_TABLE = RestypeTable(
    RESTYPES,
    RESTYPES_WITH_X,
    RESTYPE_ORDER,
    RESTYPE_ORDER_WITH_X,
)

function sequence_to_af2_indices(sequence::AbstractString; restypes::RestypeTable=RESTYPE_TABLE)
    indices = Vector{Int}(undef, lastindex(sequence))
    i = 1
    for aa in sequence
        key = string(aa)
        idx = get(restypes.restype_order_with_x, key, restypes.restype_order_with_x["X"])
        indices[i] = idx
        i += 1
    end
    return indices
end

struct Alphabet
    standard_toks::Vector{String}
    prepend_toks::Vector{String}
    append_toks::Vector{String}
    prepend_bos::Bool
    append_eos::Bool
    use_msa::Bool
    all_toks::Vector{String}
    tok_to_idx::Dict{String,Int}
    unk_idx::Int
    padding_idx::Int
    cls_idx::Int
    mask_idx::Int
    eos_idx::Int
end

const PROTEINSEQ_TOKS = [
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
    "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B",
    "U", "Z", "O", ".", "-",
]

function Alphabet(
    standard_toks::Vector{String};
    prepend_toks::Vector{String} = ["<null_0>", "<pad>", "<eos>", "<unk>"],
    append_toks::Vector{String} = ["<cls>", "<mask>", "<sep>"],
    prepend_bos::Bool = true,
    append_eos::Bool = false,
    use_msa::Bool = false,
)
    all_toks = copy(prepend_toks)
    append!(all_toks, standard_toks)
    padding_to = (8 - (length(all_toks) % 8)) % 8
    for i in 1:padding_to
        push!(all_toks, "<null_$(i)>")
    end
    append!(all_toks, append_toks)

    tok_to_idx = Dict{String,Int}(tok => i - 1 for (i, tok) in enumerate(all_toks))

    unk_idx = tok_to_idx["<unk>"]
    padding_idx = tok_to_idx["<pad>"]
    cls_idx = tok_to_idx["<cls>"]
    mask_idx = tok_to_idx["<mask>"]
    eos_idx = tok_to_idx["<eos>"]

    return Alphabet(
        standard_toks,
        prepend_toks,
        append_toks,
        prepend_bos,
        append_eos,
        use_msa,
        all_toks,
        tok_to_idx,
        unk_idx,
        padding_idx,
        cls_idx,
        mask_idx,
        eos_idx,
    )
end

function Alphabet_from_architecture(name::AbstractString)
    if name in ("ESM-1", "protein_bert_base")
        standard_toks = PROTEINSEQ_TOKS
        prepend_toks = ["<null_0>", "<pad>", "<eos>", "<unk>"]
        append_toks = ["<cls>", "<mask>", "<sep>"]
        prepend_bos = true
        append_eos = false
        use_msa = false
    elseif name in ("ESM-1b", "roberta_large")
        standard_toks = PROTEINSEQ_TOKS
        prepend_toks = ["<cls>", "<pad>", "<eos>", "<unk>"]
        append_toks = ["<mask>"]
        prepend_bos = true
        append_eos = true
        use_msa = false
    elseif name in ("MSA Transformer", "msa_transformer")
        standard_toks = PROTEINSEQ_TOKS
        prepend_toks = ["<cls>", "<pad>", "<eos>", "<unk>"]
        append_toks = ["<mask>"]
        prepend_bos = true
        append_eos = false
        use_msa = true
    else
        error("Unknown architecture: $name")
    end
    return Alphabet(
        standard_toks;
        prepend_toks=prepend_toks,
        append_toks=append_toks,
        prepend_bos=prepend_bos,
        append_eos=append_eos,
        use_msa=use_msa,
    )
end

function get_idx(alphabet::Alphabet, tok::AbstractString)
    return get(alphabet.tok_to_idx, tok, alphabet.unk_idx)
end
