import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/tmp/esm")

from esm.data import Alphabet
from esm.model.esm2 import ESM2

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

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
}


def sequence_to_af2_indices(seq: str):
    return [RESTYPE_ORDER_WITH_X.get(ch, RESTYPE_ORDER_WITH_X["X"]) for ch in seq]


class SafeTensorsReader:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._header = None
        self._base = None

    def __enter__(self):
        self._file = self.path.open("rb")
        header_len = struct.unpack("<Q", self._file.read(8))[0]
        self._header = json.loads(self._file.read(header_len))
        self._base = 8 + header_len
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    @property
    def header(self):
        return self._header

    def load(self, name: str) -> torch.Tensor:
        entry = self._header[name]
        dtype = DTYPE_MAP[entry["dtype"]]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        self._file.seek(self._base + start)
        data = self._file.read(end - start)
        tensor = torch.frombuffer(memoryview(data), dtype=dtype)
        return tensor.reshape(shape)


def infer_model_config(reader: SafeTensorsReader):
    keys = [k for k in reader.header.keys() if k != "__metadata__"]
    num_layers = max(int(k.split(".")[3]) for k in keys if k.startswith("esm.encoder.layer.")) + 1
    word = reader.load("esm.embeddings.word_embeddings.weight")
    embed_dim = word.shape[1]
    inv_freq = reader.load("esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq")
    head_dim = inv_freq.numel() * 2
    attention_heads = embed_dim // head_dim
    return num_layers, embed_dim, attention_heads


def build_reference_npz(sample_aa, sample_s_s_0):
    payload = {
        "sample_aa": sample_aa,
        "sample_s_s_0": sample_s_s_0,
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--sequence", default="ACDEFGHIK")
    parser.add_argument("--output", default="esmfold_embed_ref.npz")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    path = Path(args.safetensors)

    with SafeTensorsReader(path) as reader:
        num_layers, embed_dim, attention_heads = infer_model_config(reader)

        alphabet = Alphabet.from_architecture("ESM-1b")

        esm = ESM2(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            alphabet=alphabet,
            token_dropout=True,
        )
        esm.eval()

        # Load weights into ESM2
        state = esm.state_dict()

        state["embed_tokens.weight"].copy_(reader.load("esm.embeddings.word_embeddings.weight").float())
        state["emb_layer_norm_after.weight"].copy_(reader.load("esm.encoder.emb_layer_norm_after.weight").float())
        state["emb_layer_norm_after.bias"].copy_(reader.load("esm.encoder.emb_layer_norm_after.bias").float())

        for i in range(num_layers):
            prefix = f"esm.encoder.layer.{i}"
            state[f"layers.{i}.self_attn.q_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.query.weight").float())
            state[f"layers.{i}.self_attn.q_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.query.bias").float())
            state[f"layers.{i}.self_attn.k_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.key.weight").float())
            state[f"layers.{i}.self_attn.k_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.key.bias").float())
            state[f"layers.{i}.self_attn.v_proj.weight"].copy_(reader.load(f"{prefix}.attention.self.value.weight").float())
            state[f"layers.{i}.self_attn.v_proj.bias"].copy_(reader.load(f"{prefix}.attention.self.value.bias").float())
            state[f"layers.{i}.self_attn.out_proj.weight"].copy_(reader.load(f"{prefix}.attention.output.dense.weight").float())
            state[f"layers.{i}.self_attn.out_proj.bias"].copy_(reader.load(f"{prefix}.attention.output.dense.bias").float())

            state[f"layers.{i}.self_attn_layer_norm.weight"].copy_(reader.load(f"{prefix}.attention.LayerNorm.weight").float())
            state[f"layers.{i}.self_attn_layer_norm.bias"].copy_(reader.load(f"{prefix}.attention.LayerNorm.bias").float())

            state[f"layers.{i}.fc1.weight"].copy_(reader.load(f"{prefix}.intermediate.dense.weight").float())
            state[f"layers.{i}.fc1.bias"].copy_(reader.load(f"{prefix}.intermediate.dense.bias").float())
            state[f"layers.{i}.fc2.weight"].copy_(reader.load(f"{prefix}.output.dense.weight").float())
            state[f"layers.{i}.fc2.bias"].copy_(reader.load(f"{prefix}.output.dense.bias").float())

            state[f"layers.{i}.final_layer_norm.weight"].copy_(reader.load(f"{prefix}.LayerNorm.weight").float())
            state[f"layers.{i}.final_layer_norm.bias"].copy_(reader.load(f"{prefix}.LayerNorm.bias").float())

        esm.load_state_dict(state, strict=False)

        # Build ESMFold embedding head
        esm_s_combine = reader.load("esm_s_combine").float()

        esm_s_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, reader.load("esm_s_mlp.1.weight").shape[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(reader.load("esm_s_mlp.3.weight").shape[1], reader.load("esm_s_mlp.3.weight").shape[0]),
        )
        esm_s_mlp[0].weight.data.copy_(reader.load("esm_s_mlp.0.weight").float())
        esm_s_mlp[0].bias.data.copy_(reader.load("esm_s_mlp.0.bias").float())
        esm_s_mlp[1].weight.data.copy_(reader.load("esm_s_mlp.1.weight").float())
        esm_s_mlp[1].bias.data.copy_(reader.load("esm_s_mlp.1.bias").float())
        esm_s_mlp[3].weight.data.copy_(reader.load("esm_s_mlp.3.weight").float())
        esm_s_mlp[3].bias.data.copy_(reader.load("esm_s_mlp.3.bias").float())

        embedding_weight = reader.load("embedding.weight").float()
        embedding = torch.nn.Embedding(embedding_weight.shape[0], embedding_weight.shape[1])
        embedding.weight.data.copy_(embedding_weight)

        af2_to_esm = reader.load("af2_to_esm").long()

        seq = args.sequence
        aa = torch.tensor([sequence_to_af2_indices(seq)], dtype=torch.long)
        mask = torch.ones_like(aa)

        aa_shift = aa + 1
        aa_shift = torch.where(mask == 1, aa_shift, torch.zeros_like(aa_shift))
        esmaa = af2_to_esm[aa_shift]

        # add bos/eos as ESMFold
        bosi = alphabet.cls_idx
        eosi = alphabet.eos_idx
        pad = alphabet.padding_idx
        bos = torch.full((1, 1), bosi, dtype=torch.long)
        eos = torch.full((1, 1), pad, dtype=torch.long)
        esmaa2 = torch.cat([bos, esmaa, eos], dim=1)
        esmaa2[0, (esmaa2 != pad).sum(1)] = eosi

        res = esm(
            esmaa2,
            repr_layers=range(esm.num_layers + 1),
            need_head_weights=False,
        )

        repr0 = res["representations"][0]
        repr1 = res["representations"][1]

        esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        weights = torch.softmax(esm_s_combine, 0)
        esm_s = (weights[None, None, :, None] * esm_s).sum(2)

        s_s_0_preemb = esm_s_mlp(esm_s)
        s_s_0 = s_s_0_preemb + embedding(aa)

        config = {
            "esm_num_layers": num_layers,
            "esm_embed_dim": embed_dim,
            "esm_attention_heads": attention_heads,
            "c_s": esm_s_mlp[1].out_features,
            "c_z": None,
        }
        payload = build_reference_npz(
            aa.detach().cpu().numpy(),
            s_s_0.detach().cpu().numpy(),
        )
        payload["sample_esm_s"] = esm_s.detach().cpu().numpy()
        payload["sample_s_s_0_preemb"] = s_s_0_preemb.detach().cpu().numpy()
        payload["sample_repr0"] = repr0.detach().cpu().numpy()
        payload["sample_repr1"] = repr1.detach().cpu().numpy()

        output_path = Path(args.output)
        np.savez(output_path, **payload)
        meta = {
            "sample_sequence": seq,
            "config": config,
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta))
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
