```@meta
CurrentModule = ESMEmbed
```

# ESMEmbed

A lightweight Julia port of the ESMFold sequence embedding stack.

## Quickstart

```julia
using ESMEmbed

model = load_ESM()
emb = model("ACDEFGHIK")
```

## Outputs

The embedding output is returned in **C × L × B** order (Julia‑native layout):

- `C` = embedding width (`c_s`, typically 384)
- `L` = sequence length (after padding)
- `B` = batch size

If you request pair features with `return_pair=true` and `use_esm_attn_map=true`, the
pair tensor is returned as **C_z × L × L × B**. Otherwise `pair = nothing`.

## Input Modes

- `AbstractMatrix{Int}` shaped `(B, L)`
- `Vector{Vector{Int}}` (auto‑padded)
- `Vector{String}` or a single `String`

See the README for more usage examples.

```@index
```

```@autodocs
Modules = [ESMEmbed]
```
