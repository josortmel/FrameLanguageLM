"""Diagnostico: por que NDCG tan alto. Mira ranks, NaN y estructura temporal."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame_language_lm.data import MAX_LEN, eval_batches, load_sequences
from frame_language_lm.model import SASRec

seqs, n_items = load_sequences(
    Path("data/sequences.parquet"), Path("data/vocab_map.json"), max_users=2000
)

ckpt = torch.load("data/checkpoints/sasrec_baseline.pt", weights_only=True)
model = SASRec(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

inputs, targets, seen = eval_batches(seqs[:512], "valid", MAX_LEN)
with torch.no_grad():
    h = model(torch.from_numpy(inputs))[:, -1]
    scores = model.score_all(h)
scores[:, 0] = float("-inf")
print("NaN en scores:", torch.isnan(scores).any().item())

for i in range(len(seen)):
    scores[i, torch.from_numpy(seen[i])] = float("-inf")
tgt = torch.from_numpy(targets[:512])
tgt_scores = scores.gather(1, tgt.unsqueeze(1))
ranks = (scores > tgt_scores).sum(1).numpy()
print("rank==0:", (ranks == 0).mean(), "| mediana:", np.median(ranks))

# hipotesis leak temporal: ¿el target del valid comparte timestamp con el
# ultimo item del contexto? (sesiones de volcado masivo en ML)
import duckdb

con = duckdb.connect()
r = con.execute("""
    WITH ranked AS (
        SELECT userId, timestamp,
               row_number() OVER (PARTITION BY userId ORDER BY timestamp DESC) AS rn
        FROM read_parquet('data/sequences.parquet')
    )
    SELECT
        avg(CASE WHEN t1.timestamp = t2.timestamp THEN 1.0 ELSE 0.0 END) AS same_ts,
        avg(CASE WHEN t1.timestamp - t2.timestamp < 300 THEN 1.0 ELSE 0.0 END) AS within_5min
    FROM ranked t1 JOIN ranked t2 USING (userId)
    WHERE t1.rn = 2 AND t2.rn = 3
""").fetchone()
print(f"valid-target comparte timestamp exacto con su contexto previo: {r[0]:.1%}")
print(f"valid-target a <5min del item previo: {r[1]:.1%}")
