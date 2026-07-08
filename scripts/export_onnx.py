"""Fase 4: export ONNX del modelo de produccion (sasrec_feat).

Precomputa la matriz compuesta de items (E_ID + torre de features) y exporta
el transformer secuencia->hidden de la ultima posicion. El scoring
(hidden @ M.T) queda fuera del grafo, en numpy.

Uso: uv run python scripts/export_onnx.py [--full]
Artefactos en data/artifacts/: item_embeddings.npy, item_embeddings_int8.npz,
model_fp32.onnx, meta.json. Con --full: la tabla de embeddings del grafo es
la matriz del catalogo COMPLETO (item_embeddings_full.npy, cold-start
composicional incluido) -> model_full_fp32.onnx.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framelm.data import MAX_LEN, vocab_md5
from framelm.model import SASRec
from framelm.train import load_feature_tensors

ROOT = Path(__file__).resolve().parent.parent
CKPT = Path(
    os.environ.get("FRAMELM_CKPT", ROOT / "data/checkpoints/sasrec_feat.pt")
)
OUT = ROOT / "data/artifacts"


class ExportModel(nn.Module):
    """Transformer con la matriz compuesta congelada como buffer.

    Input: seq (B, MAX_LEN) int64, left-padded con 0.
    Output: hidden de la ultima posicion (B, d).
    """

    def __init__(self, model: SASRec, matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.pos_emb = model.pos_emb
        self.blocks = model.blocks
        self.final_ln = model.final_ln

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        pos = torch.arange(L, device=seq.device)
        h = self.matrix[seq] + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1
        )
        pad_mask = seq == 0
        keep = (~pad_mask).unsqueeze(-1)
        h = h * keep
        for blk in self.blocks:
            h = blk(h, causal, pad_mask)
            h = torch.nan_to_num(h) * keep
        return self.final_ln(h)[:, -1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="usar item_embeddings_full.npy (catalogo 100k)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, weights_only=True, map_location="cpu")
    cfg = ckpt["config"]
    assert cfg["use_features"], "se esperaba el checkpoint con features"

    feats, sizes = load_feature_tensors(
        ROOT / "data/features.npz", ROOT / "data/feature_vocabs.json"
    )
    model = SASRec(
        cfg["n_items"], d=cfg["d"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        features=feats, feature_vocab_sizes=sizes, use_rating=cfg["use_rating"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.full:
        full_path = OUT / "item_embeddings_full.npy"
        assert full_path.exists(), "genera antes scripts/build_full_matrix.py"
        matrix = torch.from_numpy(np.load(full_path)).contiguous()
    else:
        with torch.no_grad():
            matrix = model.item_matrix().contiguous()

    m = matrix.numpy().astype(np.float32)
    if not args.full:
        np.save(OUT / "item_embeddings.npy", m)
    scale = np.abs(m).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    np.savez_compressed(
        OUT / ("item_embeddings_full_int8.npz" if args.full else "item_embeddings_int8.npz"),
        q=np.round(m / scale).astype(np.int8),
        scale=scale.astype(np.float32),
    )

    export = ExportModel(model, matrix).eval()
    dummy = torch.zeros(1, MAX_LEN, dtype=torch.int64)
    dummy[0, -3:] = torch.tensor([5, 17, 42])
    fp32_path = OUT / ("model_full_fp32.onnx" if args.full else "model_fp32.onnx")
    torch.onnx.export(
        export,
        (dummy,),
        str(fp32_path),
        input_names=["seq"],
        output_names=["hidden"],
        dynamic_axes={"seq": {0: "batch"}, "hidden": {0: "batch"}},
        opset_version=17,
    )
    # NO se genera model_int8.onnx: la cuantizacion dinamica de ORT degrada
    # NDCG@10 ~70% en este modelo (medido; per_channel y excluir atencion no
    # lo arreglan) y solo ahorraria 2.5MB — la matriz de embeddings domina el
    # tamano y se distribuye aparte (fp32 + int8 propia con escala por fila).
    # Produccion: model_fp32.onnx (mediana gaps() ~5ms CPU).
    stale = OUT / "model_int8.onnx"
    if stale.exists():
        stale.unlink()

    (OUT / ("meta_full.json" if args.full else "meta.json")).write_text(
        json.dumps(
            {
                "checkpoint": CKPT.name,
                "checkpoint_epoch": ckpt["epoch"],
                "valid_ndcg@10": ckpt["valid_ndcg@10"],
                "vocab_md5": ckpt["vocab_md5"],
                "vocab_md5_actual": vocab_md5(ROOT / "data/vocab_map.json"),
                "n_items": cfg["n_items"],
                "d": cfg["d"],
                "max_len": cfg["max_len"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
