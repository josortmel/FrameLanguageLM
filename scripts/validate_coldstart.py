"""Validacion cuantificada del cold-start composicional.

A) 500 peliculas entrenadas "descabezadas" (sin E_ID): overlap de vecinos y
   NDCG@10 real (normal vs fria vs baseline popularidad).
B) Vecinos de 10 series conocidas en la matriz completa (juicio humano).
C) Secuencia real de Pepe en modo skip-unk vs composed-unk: top-10 global,
   solo series, solo documentales.

Uso: uv run python scripts/validate_coldstart.py
"""

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame_language_lm.data import eval_batches, load_sequences
from frame_language_lm.infer import FrameLM

from build_full_matrix import load_production_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_SAMPLE = 500
MAX_EVAL_USERS = 20_000

SERIES = {
    "Breaking Bad": "tt0903747",
    "The Wire": "tt0306414",
    "The Sopranos": "tt0141842",
    "True Detective": "tt2356777",
    "Dark": "tt5753856",
    "Twin Peaks": "tt0098936",
    "Chernobyl": "tt7366338",
    "The Office (US)": "tt0386676",
    "Black Mirror": "tt2085059",
    "Stranger Things": "tt4574334",
}


def unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def catalog_info() -> dict[str, tuple]:
    con = duckdb.connect(str(ROOT / "data/catalog.sqlite"), read_only=True)
    rows = con.execute(
        "SELECT tconst, primary_title, start_year, title_type, genres, num_votes "
        "FROM items"
    ).fetchall()
    con.close()
    return {t: (title, year, typ, gen or "", nv or 0) for t, title, year, typ, gen, nv in rows}


@torch.no_grad()
def ndcg_at_10(model, matrix, inputs, targets, seen, batch_size=512) -> float:
    total = 0.0
    for s in range(0, len(targets), batch_size):
        e = min(s + batch_size, len(targets))
        x = torch.from_numpy(inputs[s:e])
        h = model(x, matrix=matrix)[:, -1]
        scores = h @ matrix.T
        scores[:, 0] = float("-inf")
        rows = np.concatenate([np.full(len(seen[s + i]), i) for i in range(e - s)])
        cols = np.concatenate([seen[s + i] for i in range(e - s)])
        scores[torch.from_numpy(rows), torch.from_numpy(cols)] = float("-inf")
        tgt = torch.from_numpy(targets[s:e]).unsqueeze(1)
        ranks = (scores > scores.gather(1, tgt)).sum(1)
        total += (1.0 / torch.log2(ranks.float() + 2.0))[ranks < 10].sum().item()
    return total / len(targets)


def ndcg_popularity(pop, targets, seen) -> float:
    total = 0.0
    for i, tgt in enumerate(targets):
        p = pop.copy()
        p[0] = -np.inf
        p[seen[i]] = -np.inf
        rank = int((p > p[tgt]).sum())
        if rank < 10:
            total += 1.0 / np.log2(rank + 2.0)
    return total / len(targets)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    info = catalog_info()
    mapping = json.loads(
        (ROOT / "data/vocab_map.json").read_text(encoding="utf-8")
    )["tconst_to_idx"]
    idx_to_tconst = {i: t for t, i in mapping.items()}
    full_map = json.loads(
        (ROOT / "data/full_vocab_map.json").read_text(encoding="utf-8")
    )["tconst_to_idx"]
    full_idx_to_tconst = {i: t for t, i in full_map.items()}

    def describe(tconst: str) -> str:
        title, year, typ, _, _ = info.get(tconst, ("?", None, "?", "", 0))
        tag = "" if typ == "movie" else f" [{typ}]"
        return f"{title} ({year}){tag}"

    # ---------- EXPERIMENTO A ----------
    if args.only in (None, "A"):
        print("=== A: calidad de la composicion (500 peliculas descabezadas) ===")
        seqs, n_items = load_sequences(
            ROOT / "data/sequences.parquet", ROOT / "data/vocab_map.json"
        )
        counts = np.bincount(np.concatenate(seqs), minlength=n_items + 1)
        is_movie = np.zeros(n_items + 1, dtype=bool)
        pop = np.zeros(n_items + 1, dtype=np.float64)
        for i in range(1, n_items + 1):
            t = idx_to_tconst[i]
            _, _, typ, _, nv = info.get(t, (None, None, "?", "", 0))
            is_movie[i] = typ == "movie"
            pop[i] = nv
        cand = np.where((counts >= 100) & is_movie)[0]
        sel = rng.choice(cand, N_SAMPLE, replace=False)

        model = load_production_model()
        with torch.no_grad():
            matrix = model.item_matrix().contiguous()
            tower = model.tower()
        cold_matrix = matrix.clone()
        cold_matrix[sel] = tower[sel]

        m_np = matrix.numpy()
        u = unit(m_np)
        sims_t = u[sel] @ u.T
        sims_c = unit(tower[sel].numpy()) @ u.T
        for a, row in enumerate(sel):
            sims_t[a, row] = -np.inf
            sims_c[a, row] = -np.inf
        sims_t[:, 0] = -np.inf
        sims_c[:, 0] = -np.inf
        top_t = np.argsort(sims_t, axis=1)[:, -10:]
        top_c = np.argsort(sims_c, axis=1)[:, -10:]
        overlap = np.mean(
            [len(set(top_t[a]) & set(top_c[a])) / 10.0 for a in range(len(sel))]
        )
        print(f"overlap@10 vecinos (fria vs entrenada): {overlap:.3f}")

        inputs, targets, seen = eval_batches(seqs, "valid")
        keep = np.where(np.isin(targets, sel))[0]
        if len(keep) > MAX_EVAL_USERS:
            keep = rng.choice(keep, MAX_EVAL_USERS, replace=False)
        inputs_k = inputs[keep]
        targets_k = targets[keep]
        seen_k = [seen[i] for i in keep]
        print(f"usuarios de valid con target en la muestra: {len(keep):,}")

        ndcg_normal = ndcg_at_10(model, matrix, inputs_k, targets_k, seen_k)
        ndcg_cold = ndcg_at_10(model, cold_matrix, inputs_k, targets_k, seen_k)
        ndcg_pop = ndcg_popularity(pop, targets_k, seen_k)
        print(f"NDCG@10 normal:      {ndcg_normal:.4f}")
        print(f"NDCG@10 fria:        {ndcg_cold:.4f}  ({ndcg_cold/ndcg_normal:.1%} del normal)")
        print(f"NDCG@10 popularidad: {ndcg_pop:.4f}  (fria = {ndcg_cold/max(ndcg_pop,1e-9):.1f}x popularidad)")

    # ---------- EXPERIMENTO B ----------
    if args.only in (None, "B"):
        print("\n=== B: vecinos de series (matriz completa, juicio humano) ===")
        full = np.load(ROOT / "data/artifacts/item_embeddings_full.npy")
        uf = unit(full)
        for name, t in SERIES.items():
            row = full_map.get(t)
            if row is None:
                print(f"\n{name}: NO esta en el catalogo top-100k")
                continue
            sims = uf @ uf[row]
            sims[0] = -np.inf
            sims[row] = -np.inf
            top = np.argsort(sims)[-10:][::-1]
            kind = "entrenada" if row <= len(mapping) else "COLD"
            print(f"\n{name} [{kind}]")
            for j in top:
                print(f"  {sims[j]:.3f}  {describe(full_idx_to_tconst[int(j)])}")

    # ---------- EXPERIMENTO C ----------
    if args.only not in (None, "C"):
        return
    print("\n=== C: secuencia de Pepe, skip-unk vs composed-unk ===")
    lm = FrameLM()
    assert lm.full, "FrameLM debe estar usando los artefactos full"
    con = duckdb.connect()
    cols = [
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{ROOT / 'data/user/pepe_sequence.parquet'}')"
        ).fetchall()
    ]
    icol = "indice" if "indice" in cols else "idx"
    rows_p = con.execute(
        f"SELECT tconst, {icol} FROM read_parquet('{ROOT / 'data/user/pepe_sequence.parquet'}') "
        "ORDER BY timestamp"
    ).fetchall()
    con.close()
    skip_seq = [int(i) for _, i in rows_p if int(i) > 0]
    composed_seq = [int(full_map[t]) for t, _ in rows_p if t in full_map]
    seen_full = set(composed_seq)
    print(f"eventos: skip={len(skip_seq)}, composed={len(composed_seq)}")

    n_full = lm.matrix.shape[0]
    series_mask = np.zeros(n_full, dtype=bool)
    docs_mask = np.zeros(n_full, dtype=bool)
    for t, row in full_map.items():
        _, _, typ, gen, _ = info.get(t, (None, None, "?", "", 0))
        if typ != "movie":
            series_mask[row] = True
        if "Documentary" in gen:
            docs_mask[row] = True

    def show(tag, seq):
        for label, mask in (("GLOBAL", None), ("SERIES", series_mask), ("DOCUMENTALES", docs_mask)):
            out = lm.gaps(seq, k=10, exclude=seen_full, mask=mask)
            print(f"\n[{tag}] top-10 {label}:")
            for g in out:
                row = full_map.get(g["tconst"], 0)
                cold = "*" if row > len(mapping) else " "
                print(f" {cold} {g['score']:.2f}  {describe(g['tconst'])}")

    show("SKIP-UNK", skip_seq)
    show("COMPOSED-UNK", composed_seq)
    print("\n(* = item cold-start, embedding solo composicional)")


if __name__ == "__main__":
    main()
