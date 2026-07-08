"""Cold-start: compone la matriz de items para el catalogo COMPLETO (100k).

Filas 1..n_train = matriz entrenada (E_ID + torre). Filas n_train+1.. =
solo proj(features) con los embeddings de feature YA entrenados (E_ID no
existe para items sin señal colaborativa). Fila 0 = <pad> = 0.

Artefactos: data/artifacts/item_embeddings_full.npy,
data/full_vocab_map.json, data/artifacts/full_meta.json

Uso: uv run python scripts/build_full_matrix.py
"""

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framelm.features import (
    N_CAST,
    N_DIRECTORS,
    N_GENRES,
    _split,
    budget_bucket,
    decade_bucket,
)
from framelm.model import SASRec
from framelm.train import load_feature_tensors

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "data/checkpoints/sasrec_feat.pt"


def load_production_model() -> SASRec:
    ckpt = torch.load(CKPT, weights_only=True, map_location="cpu")
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
    return model


def feature_indices_for(rows: list, vocabs: dict) -> dict[str, torch.Tensor]:
    """Indices de feature para items nuevos usando vocabularios YA entrenados.

    Nombre fuera de vocabulario -> 0 (desconocido).
    """
    n = len(rows)
    dir_idx = np.zeros((n, N_DIRECTORS), dtype=np.int64)
    cast_idx = np.zeros((n, N_CAST), dtype=np.int64)
    gen_idx = np.zeros((n, N_GENRES), dtype=np.int64)
    country = np.zeros(n, dtype=np.int64)
    language = np.zeros(n, dtype=np.int64)
    decade = np.zeros(n, dtype=np.int64)
    budget = np.zeros(n, dtype=np.int64)

    for i, (_, dirs, cast, gens, ctry, lang, year, budg) in enumerate(rows):
        for j, name in enumerate(_split(dirs, "|", N_DIRECTORS)):
            dir_idx[i, j] = vocabs["director"].get(name, 0)
        for j, name in enumerate(_split(cast, "|", N_CAST)):
            cast_idx[i, j] = vocabs["cast"].get(name, 0)
        for j, g in enumerate(_split(gens, ",", N_GENRES)):
            gen_idx[i, j] = vocabs["genre"].get(g, 0)
        c = _split(ctry, "|", 1)
        if c:
            country[i] = vocabs["country"].get(c[0], 0)
        if lang:
            language[i] = vocabs["language"].get(lang, 0)
        decade[i] = decade_bucket(year)
        budget[i] = budget_bucket(budg)

    return {
        "director": torch.from_numpy(dir_idx),
        "cast": torch.from_numpy(cast_idx),
        "genre": torch.from_numpy(gen_idx),
        "country": torch.from_numpy(country).unsqueeze(1),
        "language": torch.from_numpy(language).unsqueeze(1),
        "decade": torch.from_numpy(decade).unsqueeze(1),
        "budget": torch.from_numpy(budget).unsqueeze(1),
    }


@torch.no_grad()
def compose_tower_term(model: SASRec, idx: dict[str, torch.Tensor]) -> torch.Tensor:
    """proj(concat(mean(emb(feature)))) para filas arbitrarias — sin E_ID."""
    parts = []
    for key in ("director", "cast", "genre", "country", "language", "decade", "budget"):
        emb = getattr(model.tower, f"emb_{key}")(idx[key])
        valid = (idx[key] > 0).unsqueeze(-1).float()
        denom = valid.sum(1).clamp(min=1.0)
        parts.append((emb * valid).sum(1) / denom)
    return model.tower.proj(torch.cat(parts, dim=-1))


def main() -> None:
    mapping = json.loads(
        (ROOT / "data/vocab_map.json").read_text(encoding="utf-8")
    )["tconst_to_idx"]
    n_train = len(mapping)
    vocabs = json.loads(
        (ROOT / "data/feature_vocabs.json").read_text(encoding="utf-8")
    )

    con = duckdb.connect(str(ROOT / "data/catalog.sqlite"), read_only=True)
    rows = con.execute(
        """SELECT tconst, directors, "cast", genres, countries,
           original_language, start_year, budget
           FROM items ORDER BY num_votes DESC, tconst"""
    ).fetchall()
    con.close()

    new_rows = [r for r in rows if r[0] not in mapping]
    model = load_production_model()

    with torch.no_grad():
        trained = model.item_matrix().numpy().astype(np.float32)  # (n_train+1, d)

    idx = feature_indices_for(new_rows, vocabs)
    cold = compose_tower_term(model, idx).numpy().astype(np.float32)

    full = np.concatenate([trained, cold], axis=0)
    out_dir = ROOT / "data/artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_embeddings_full.npy", full)

    full_map = dict(mapping)
    for offset, r in enumerate(new_rows):
        full_map[r[0]] = n_train + 1 + offset
    (ROOT / "data/full_vocab_map.json").write_text(
        json.dumps({"n_items": len(full_map), "tconst_to_idx": full_map}),
        encoding="utf-8",
    )

    # cobertura de features de los nuevos: cuantos tienen ALGO de señal
    known_dir = (idx["director"] > 0).any(1).float().mean().item()
    known_cast = (idx["cast"] > 0).any(1).float().mean().item()
    known_any = (
        (idx["director"] > 0).any(1)
        | (idx["cast"] > 0).any(1)
        | (idx["genre"] > 0).any(1)
    ).float().mean().item()
    meta = {
        "n_train": n_train,
        "n_cold": len(new_rows),
        "n_total": len(full_map),
        "cold_con_director_conocido": round(known_dir, 4),
        "cold_con_cast_conocido": round(known_cast, 4),
        "cold_con_alguna_feature": round(known_any, 4),
    }
    (out_dir / "full_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
