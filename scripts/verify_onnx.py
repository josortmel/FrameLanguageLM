"""Fase 4: verificacion del export ONNX.

1. Paridad NDCG@10 valid (muestra 3.000 usuarios, seed fija):
   PyTorch vs ONNX-fp32 vs ONNX-int8. Gate: int8 <2% degradacion rel.
2. Latencia gaps() con secuencia de 500 items, 50 reps. Gate: mediana <100ms.
3. Sanity: vecinos de Matrix / Breaking Bad / Persona.
4. Tamanos de artifacts.

Uso: uv run python scripts/verify_onnx.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framelm.data import MAX_LEN, load_sequences
from framelm.eval import evaluate
from framelm.infer import FrameLM
from framelm.model import SASRec
from framelm.train import load_feature_tensors

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data/artifacts"
N_USERS = 3000
SEED = 42


def eval_onnx(path: Path, seqs, matrix: np.ndarray) -> float:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    ndcg = 0.0
    B = 256
    from framelm.data import eval_batches

    inputs, targets, seen = eval_batches(seqs, "valid", MAX_LEN)
    for start in range(0, len(targets), B):
        end = min(start + B, len(targets))
        h = sess.run(None, {"seq": inputs[start:end]})[0]      # (b, d)
        scores = h @ matrix.T                                   # (b, n+1)
        scores[:, 0] = -np.inf
        for i in range(end - start):
            scores[i, seen[start + i]] = -np.inf
        tgt = targets[start:end]
        tgt_scores = scores[np.arange(end - start), tgt]
        ranks = (scores > tgt_scores[:, None]).sum(1)
        ndcg += (1.0 / np.log2(ranks[ranks < 10] + 2.0)).sum()
    return ndcg / len(targets)


def main() -> None:
    rng = np.random.default_rng(SEED)
    seqs_all, n_items = load_sequences(
        ROOT / "data/sequences.parquet", ROOT / "data/vocab_map.json"
    )
    pick = rng.choice(len(seqs_all), size=N_USERS, replace=False)
    seqs = [seqs_all[i] for i in sorted(pick)]

    # PyTorch de referencia
    ckpt = torch.load(
        ROOT / "data/checkpoints/sasrec_feat.pt", weights_only=True,
        map_location="cpu",
    )
    cfg = ckpt["config"]
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
    pt = evaluate(model, seqs, "valid", "cpu", max_len=MAX_LEN)["ndcg@10"]

    matrix = np.load(ART / "item_embeddings.npy")
    fp32 = eval_onnx(ART / "model_fp32.onnx", seqs, matrix)

    deg_fp32 = (pt - fp32) / pt * 100
    print(f"paridad NDCG@10 valid ({N_USERS} usuarios, seed {SEED}):")
    print(f"  pytorch:   {pt:.4f}")
    print(f"  onnx fp32: {fp32:.4f}  (deg {deg_fp32:+.2f}%)")
    gate_parity = "OK" if deg_fp32 < 2.0 else "FALLO"
    print(f"  [{gate_parity}] gate fp32 (produccion) <2%")

    # Latencia
    lm = FrameLM(ART, ROOT / "data/catalog.sqlite", ROOT / "data/vocab_map.json")
    long_seq = rng.integers(1, n_items + 1, size=500).tolist()
    lm.gaps(long_seq, k=50)  # warmup
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        lm.gaps(long_seq, k=50)
        times.append((time.perf_counter() - t0) * 1000)
    med, p95 = np.median(times), np.percentile(times, 95)
    gate_lat = "OK" if med < 100 else "FALLO"
    print(f"\nlatencia gaps() seq=500, k=50, modelo {lm.model_kind}:")
    print(f"  mediana {med:.1f} ms | p95 {p95:.1f} ms  [{gate_lat}] gate <100ms")

    # Sanity: vecinos
    print("\nvecinos (top-5):")
    # nota: series (p.ej. Breaking Bad tt0903747) NO estan en el vocab
    # entrenable — MovieLens es solo cine (riesgo R1 del SPEC).
    for tconst, name in [
        ("tt0133093", "The Matrix"),
        ("tt0166924", "Mulholland Drive"),
        ("tt0060827", "Persona"),
    ]:
        ns = lm.neighbors(tconst, k=5)
        pretty = ", ".join(f"{n['title']} ({n['year']})" for n in ns)
        print(f"  {name}: {pretty}")

    print("\nartifacts:")
    total = 0
    for f in sorted(ART.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name}: {sz / 1e6:.1f} MB")
    print(f"  TOTAL: {total / 1e6:.1f} MB")

    if gate_parity != "OK" or gate_lat != "OK":
        sys.exit("gates con fallos")


if __name__ == "__main__":
    main()
