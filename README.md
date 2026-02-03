# ESMEmbed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ESMEmbed.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ESMEmbed.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ESMEmbed.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ESMEmbed.jl/actions/workflows/CI.yml?query=branch%3Amain)

A lightweight Julia port of the **ESMFold sequence embedding stack**. This package lets you load
ESMFold weights (from Hugging Face) and compute **per‑residue embeddings** on CPU.
It does **not** include the ESMFold structure module.

## Quickstart

```julia
using ESMEmbed

# Download weights from Hugging Face and build the model
model = load_ESM()

# Single sequence
emb = model("ACDEFGHIK")

# Batch of sequences (auto‑padding + mask)
emb_batch = model(["ACDEFGHIK", "MKT"])
```

## What The Outputs Are

The main call returns **per‑residue sequence embeddings** (the inputs to the structure module in ESMFold).
For Julia‑native layout, tensors are returned in **C × L × B** order:

- `emb` has shape `(c_s, L, B)`
  - `c_s`: embedding width (from the checkpoint; typically 384)
  - `L`: sequence length (after padding)
  - `B`: batch size

If you want both sequence and pair features:

```julia
out = model(["ACDEFGHIK"]; return_pair=true)
seq = out.sequence   # (c_s, L, B)
pair = out.pair      # (c_z, L, L, B)
```

`pair` is only produced when `use_esm_attn_map=true` (see below). Otherwise it is `nothing`.

## Input Conveniences

You can pass any of the following:

- `AbstractMatrix{Int}` shaped `(B, L)`
- `Vector{Vector{Int}}` (auto‑padded, mask auto‑generated)
- `Vector{String}` or a single `String`

Indices are **AF2 restype indices** (0‑based). Use:

```julia
seq_ints = sequence_to_af2_indices("ACDEFGHIK")
```

## Weights And Caching

`load_ESM()` downloads the safetensors checkpoint from Hugging Face using
`HuggingFaceApi.hf_hub_download`. By default it pulls:

- `repo_id = "facebook/esmfold_v1"`
- `filename = "model.safetensors"`
- `revision = "ba837a3"`

Downloaded files are cached by HuggingFaceApi in your Julia depot (via OhMyArtifacts).
You can override the source if you want to point at a PR or a specific commit:

```julia
model = load_ESM(
    repo_id = "facebook/esmfold_v1",
    filename = "esm.safetensors",
    revision = "refs/pr/123",
)
```

You can also skip network access and use the local cache only:

```julia
model = load_ESM(local_files_only=true)
```

## Advanced Usage

### Pre‑padded batch with mask

```julia
aa = [
    0 1 2 3 4 5;
    0 1 2 0 0 0;
]
mask = [
    1 1 1 1 1 1;
    1 1 1 0 0 0;
]
emb = model(aa; mask=mask)
```

### Pair Features

```julia
model = load_ESM(use_esm_attn_map=true)
out = model(["ACDEFGHIK"]; return_pair=true)
```

## Notes

- CPU‑only execution is supported.
- The implementation follows the ESM2 and ESMFold embedding pathway closely, with parity
  against the original Python model to small floating‑point tolerances.

## License

This package reuses ESM code concepts and weight formats. Please refer to the original
ESM/ESMFold licenses and terms for model usage.
