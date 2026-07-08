"""Cold-start: compone la matriz de items para el catalogo COMPLETO (100k).

Filas 1..n_train = matriz entrenada (E_ID + torre). Filas n_train+1.. =
solo proj(features) con los embeddings de feature YA entrenados (E_ID no
existe para items sin señal colaborativa). Fila 0 = <pad> = 0.

Artefactos: data/artifacts/item_embeddings_full.npy,
data/full_vocab_map.json, data/artifacts/full_meta.json

Uso: uv run python scripts/build_full_matrix.py
"""

import json
import os
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
CKPT = Path(
    os.environ.get("FRAMELM_CKPT", ROOT / "data/checkpoints/sasrec_feat.pt")
)


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


def series_writers(series_tconsts: set[str]) -> dict[str, list[str]]:
    """F4: primeros 3 writers (nombres) por serie, cacheado en interim."""
    cache = ROOT / "data/interim/series_writers.parquet"
    con = duckdb.connect()
    if not cache.exists():
        raw = ROOT / "data" / "raw"
        con.execute(f"""
            COPY (
                WITH crew AS (
                    SELECT tconst, string_split(writers, ',') AS ws
                    FROM read_csv_auto('{raw / "title.crew.tsv.gz"}',
                                       delim='\t', quote='', nullstr='\\N')
                    WHERE writers IS NOT NULL
                ), x AS (
                    SELECT tconst, unnest(ws[1:3]) AS nconst FROM crew
                )
                SELECT x.tconst, list(n.primaryName) AS writer_names
                FROM x JOIN read_csv_auto('{raw / "name.basics.tsv.gz"}',
                                          delim='\t', quote='', nullstr='\\N') n
                  ON x.nconst = n.nconst
                GROUP BY x.tconst
            ) TO '{cache}' (FORMAT PARQUET)
        """)
    rows = con.execute(f"SELECT tconst, writer_names FROM '{cache}'").fetchall()
    con.close()
    return {t: ws for t, ws in rows if t in series_tconsts}


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
    cat_info = {
        t: (tt, g or "", int(nv or 0))
        for t, tt, g, nv in con.execute(
            "SELECT tconst, title_type, genres, num_votes FROM items"
        ).fetchall()
    }
    con.close()

    new_rows = [r for r in rows if r[0] not in mapping]

    # F4: para series frias, slots de director <- writers/creadores si alguno
    # cae en el vocab de directores entrenado (19% de las series; medido)
    series_t = {r[0] for r in new_rows if cat_info[r[0]][0] != "movie"}
    writers = series_writers(series_t)
    dir_vocab = vocabs["director"]
    swapped = 0
    for i, r in enumerate(new_rows):
        ws = writers.get(r[0])
        if ws and any(w in dir_vocab for w in ws):
            known = [w for w in ws if w in dir_vocab]
            new_rows[i] = (r[0], "|".join(known[:3]), *r[2:])
            swapped += 1
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

    # aux por fila (alineado a la matriz completa): flags para filtrado y
    # prior de popularidad en el ranking frio. Fila 0 = <pad>.
    n_total_rows = full.shape[0]
    trained_feats = np.load(ROOT / "data/features.npz")
    director_known = np.zeros(n_total_rows, dtype=bool)
    cast_known_n = np.zeros(n_total_rows, dtype=np.int8)
    num_votes = np.zeros(n_total_rows, dtype=np.int64)
    is_movie = np.zeros(n_total_rows, dtype=bool)
    is_series = np.zeros(n_total_rows, dtype=bool)
    is_doc = np.zeros(n_total_rows, dtype=bool)
    is_cold = np.zeros(n_total_rows, dtype=bool)
    is_cold[n_train + 1:] = True

    director_known[: n_train + 1] = (trained_feats["director"] > 0).any(1)
    cast_known_n[: n_train + 1] = (trained_feats["cast"] > 0).sum(1)
    director_known[n_train + 1:] = (idx["director"] > 0).any(1).numpy()
    cast_known_n[n_train + 1:] = (idx["cast"] > 0).sum(1).numpy()

    for t, row in full_map.items():
        tt, genres, nv = cat_info[t]
        num_votes[row] = nv
        is_movie[row] = tt == "movie"
        is_series[row] = tt != "movie"
        is_doc[row] = "Documentary" in genres

    np.savez_compressed(
        out_dir / "full_aux.npz",
        director_known=director_known, cast_known_n=cast_known_n,
        num_votes=num_votes, is_movie=is_movie, is_series=is_series,
        is_doc=is_doc, is_cold=is_cold,
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
        "checkpoint": CKPT.name,
        "n_train": n_train,
        "n_cold": len(new_rows),
        "n_total": len(full_map),
        "series_con_writers_activados": swapped,
        "cold_con_director_conocido": round(known_dir, 4),
        "cold_con_cast_conocido": round(known_cast, 4),
        "cold_con_alguna_feature": round(known_any, 4),
    }
    (out_dir / "full_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
