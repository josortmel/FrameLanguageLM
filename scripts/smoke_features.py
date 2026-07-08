"""Smoke test fase 3: features composicionales + rating.

Uso: uv run python scripts/smoke_features.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framelm.data import rating_bucket  # noqa: E402
from framelm.loss import gbce_loss  # noqa: E402
from framelm.model import FEATURE_KEYS, SASRec  # noqa: E402
from framelm.train import load_feature_tensors  # noqa: E402

torch.manual_seed(7)

# --- rating buckets ---
rb = rating_bucket(np.array([0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
assert rb.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4], rb
print("[OK] rating_bucket")

# --- features reales alineadas con el vocab ---
feats, sizes = load_feature_tensors(
    Path("data/features.npz"), Path("data/feature_vocabs.json")
)
n_items = json.loads(Path("data/vocab_map.json").read_text(encoding="utf-8"))["n_items"]
for k in FEATURE_KEYS:
    arr = feats[k]
    assert arr.shape[0] == n_items + 1, (k, arr.shape)
    assert int(arr[0].max()) == 0, f"fila <pad> de {k} debe ser 0"
print(f"[OK] features.npz alineado ({n_items:,} items)")

# --- embedding cambia con/sin features ---
m_id = SASRec(n_items)
m_ft = SASRec(n_items, features=feats, feature_vocab_sizes=sizes)
with torch.no_grad():
    m_ft.item_emb.weight.copy_(m_id.item_emb.weight)
    base = m_id.item_matrix()
    comp = m_ft.item_matrix()
# item con features (idx 1 existe): su fila compuesta debe diferir del solo-ID
assert not torch.allclose(base[1], comp[1]), "features no afectan al embedding"
assert torch.all(comp[0] == 0), "fila <pad> compuesta debe ser 0"
print("[OK] embedding compuesto difiere del solo-ID y <pad> queda a 0")

# --- forward + loss con rating y filtro de targets, sin NaN ---
model = SASRec(n_items, features=feats, feature_vocab_sizes=sizes, use_rating=True)
B, L = 4, 16
x = torch.randint(1, n_items + 1, (B, L))
x[0, :10] = 0  # padding parcial
y = torch.randint(1, n_items + 1, (B, L))
rx = torch.randint(0, 5, (B, L))
y_ok = torch.rand(B, L) > 0.4
matrix = model.item_matrix()
h = model(x, ratings=rx, matrix=matrix)
assert not torch.isnan(h).any(), "NaN en hidden states"
loss = gbce_loss(h, y, matrix, n_items, k=32, target_mask=y_ok)
assert torch.isfinite(loss), loss
loss.backward()
print(f"[OK] forward+loss+backward con features/rating (loss={loss.item():.4f})")

# --- batch con TODOS los targets filtrados no produce NaN ---
loss0 = gbce_loss(h.detach(), y, matrix.detach(), n_items, k=32,
                  target_mask=torch.zeros(B, L, dtype=torch.bool))
assert loss0.item() == 0.0, loss0
print("[OK] batch 100% filtrado -> loss 0, sin NaN")

print("\nsmoke features: 5/5 verde")
