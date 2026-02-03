import argparse
import json
from pathlib import Path

import numpy as np
import torch

import esm

RESTYPES = [
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
RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER_WITH_X = {aa: i for i, aa in enumerate(RESTYPES_WITH_X)}


def sequence_to_af2_indices(seq: str):
    return [RESTYPE_ORDER_WITH_X.get(ch, RESTYPE_ORDER_WITH_X["X"]) for ch in seq]


def build_state_npz(state_dict, extra_arrays=None):
    names = []
    arrays = []
    for name, tensor in state_dict.items():
        names.append(name)
        arrays.append(tensor.detach().cpu().numpy())

    if extra_arrays:
        for name, arr in extra_arrays.items():
            names.append(name)
            arrays.append(arr)

    payload = {f"arr_{i+1}": arr for i, arr in enumerate(arrays)}
    return payload, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="esmfold_structure_module_only_8M")
    parser.add_argument("--sequence", default="ACDEFGHIKLMNPQRSTVWY")
    parser.add_argument("--output", default="esmfold_embed_8m.npz")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Load model
    if args.model.endswith(".pt"):
        model = esm.esmfold.v1.pretrained._load_model(args.model)
    else:
        model = getattr(esm.pretrained, args.model)()
    model.eval()
    model.float()

    seq = args.sequence
    aa = torch.tensor([sequence_to_af2_indices(seq)], dtype=torch.long)
    mask = torch.ones_like(aa)

    esmaa = model._af2_idx_to_esm_idx(aa, mask)
    esm_s, _ = model._compute_language_model_representations(esmaa)
    esm_s = esm_s.to(model.esm_s_combine.dtype).detach()
    esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

    s_s_0 = model.esm_s_mlp(esm_s)
    s_s_0 = s_s_0 + model.embedding(aa)

    s_s_0_np = s_s_0.detach().cpu().numpy()

    state = model.state_dict()
    config = {
        "esm_num_layers": model.esm.num_layers,
        "esm_embed_dim": model.esm.embed_dim,
        "esm_attention_heads": model.esm.attention_heads,
        "c_s": model.esm_s_mlp[1].out_features,
        "c_z": getattr(model.cfg.trunk, "pairwise_state_dim", None),
    }
    extra_arrays = {
        "sample_aa": aa.detach().cpu().numpy(),
        "sample_s_s_0": s_s_0_np,
    }
    payload, names = build_state_npz(state, extra_arrays=extra_arrays)

    output_path = Path(args.output)
    np.savez(output_path, **payload)
    meta = {
        "names": names,
        "sample_sequence": seq,
        "config": config,
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".names.json")
    meta_path.write_text(json.dumps(meta))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
