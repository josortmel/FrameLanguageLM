"""Carga de secuencias y split leave-one-out temporal.

Vocabulario: tconst -> indice contiguo (1..n_items), 0 reservado para <pad>.
Se construye SIEMPRE sobre el parquet completo (estable entre smoke tests y
entrenamiento completo) y se persiste en data/vocab_map.json.
"""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 0
MAX_LEN = 200


def _vocab_query(parquet: str) -> str:
    # orden deterministico: frecuencia desc, tconst como desempate
    return f"""
        SELECT tconst, row_number() OVER (ORDER BY cnt DESC, tconst) AS idx
        FROM (SELECT tconst, count(*) AS cnt
              FROM read_parquet('{parquet}') GROUP BY tconst)
    """


def build_or_load_vocab(parquet: Path, vocab_path: Path) -> dict[str, int]:
    if vocab_path.exists():
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        return payload["tconst_to_idx"]
    con = duckdb.connect()
    rows = con.execute(_vocab_query(str(parquet))).fetchall()
    mapping = {t: i for t, i in rows}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps({"n_items": len(mapping), "tconst_to_idx": mapping}),
        encoding="utf-8",
    )
    return mapping


def vocab_md5(vocab_path: Path) -> str:
    return hashlib.md5(vocab_path.read_bytes()).hexdigest()


def rating_bucket(ratings: np.ndarray) -> np.ndarray:
    """0=sin rating, 1=<=2, 2=2.5-3, 3=3.5-4, 4=4.5-5."""
    out = np.zeros(len(ratings), dtype=np.int64)
    valid = ratings > 0
    out[valid & (ratings <= 2.0)] = 1
    out[valid & (ratings > 2.0) & (ratings <= 3.0)] = 2
    out[valid & (ratings > 3.0) & (ratings <= 4.0)] = 3
    out[valid & (ratings > 4.0)] = 4
    return out


def load_sequences(
    parquet: Path,
    vocab_path: Path,
    max_users: int | None = None,
    with_ratings: bool = False,
):
    """Devuelve (secuencias por usuario en orden temporal, n_items).

    Con with_ratings=True devuelve (seqs, rating_seqs, n_items), con
    rating_seqs alineado 1:1 con seqs (valores crudos 0.5-5.0).
    """
    mapping = build_or_load_vocab(parquet, vocab_path)
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE vocab(tconst VARCHAR, idx BIGINT)")
    con.executemany("INSERT INTO vocab VALUES (?, ?)", mapping.items())
    user_filter = ""
    if max_users:
        user_filter = f"""
            WHERE userId IN (SELECT userId FROM (
                SELECT DISTINCT userId FROM read_parquet('{parquet}')
                ORDER BY userId LIMIT {max_users}))
        """
    rating_col = ", s.rating" if with_ratings else ""
    arr = con.execute(f"""
        SELECT s.userId, v.idx{rating_col}
        FROM read_parquet('{parquet}') s JOIN vocab v USING (tconst)
        {user_filter}
        ORDER BY s.userId, s.timestamp
    """).fetchnumpy()
    idx = np.ascontiguousarray(arr["idx"], dtype=np.int64)
    ratings = (
        np.ascontiguousarray(arr["rating"], dtype=np.float32)
        if with_ratings
        else np.zeros(len(idx), dtype=np.float32)
    )
    _, starts = np.unique(arr["userId"], return_index=True)
    bounds = np.append(starts, len(idx))
    pairs = [
        (idx[bounds[i] : bounds[i + 1]], ratings[bounds[i] : bounds[i + 1]])
        for i in range(len(starts))
    ]
    # split leave-one-out necesita >=3 items (train>=1, valid, test)
    pairs = [(s, r) for s, r in pairs if len(s) >= 3]
    seqs = [s for s, _ in pairs]
    if with_ratings:
        return seqs, [r for _, r in pairs], len(mapping)
    return seqs, len(mapping)


def left_pad(window: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    out[length - len(window) :] = window
    return out


class TrainDataset(Dataset):
    """Una muestra por usuario: ventana de los ultimos max_len+1 items de train.

    Con ratings devuelve ademas (rx, y_ok): buckets de rating del INPUT y
    mascara de targets validos (rating del target >= min_target_rating).
    """

    def __init__(
        self,
        seqs: list[np.ndarray],
        max_len: int = MAX_LEN,
        ratings: list[np.ndarray] | None = None,
        min_target_rating: float = 3.5,
    ):
        self.max_len = max_len
        self.min_target_rating = min_target_rating
        keep = [i for i, s in enumerate(seqs) if len(s) >= 4]
        # train = s[:-2]; hace falta len>=2 ahi para tener al menos un par (x, y)
        self.train_seqs = [seqs[i][:-2] for i in keep]
        self.train_ratings = (
            [ratings[i][:-2] for i in keep] if ratings is not None else None
        )

    def __len__(self) -> int:
        return len(self.train_seqs)

    def __getitem__(self, i: int):
        s = self.train_seqs[i]
        window = s[-(self.max_len + 1) :]
        x = left_pad(window[:-1], self.max_len)
        y = left_pad(window[1:], self.max_len)
        if self.train_ratings is None:
            return torch.from_numpy(x), torch.from_numpy(y)
        r = self.train_ratings[i][-(self.max_len + 1) :]
        rx = left_pad(rating_bucket(r[:-1]), self.max_len)
        y_rating = np.zeros(self.max_len, dtype=np.float32)
        y_rating[self.max_len - len(r) + 1 :] = r[1:]
        y_ok = torch.from_numpy(y_rating >= self.min_target_rating)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(rx), y_ok


def eval_batches(
    seqs: list[np.ndarray],
    mode: str,
    max_len: int = MAX_LEN,
    ratings: list[np.ndarray] | None = None,
):
    """(inputs (U, max_len), targets (U,), seen por usuario) para valid o test.

    Con ratings devuelve ademas input_ratings (U, max_len) en buckets.
    """
    assert mode in ("valid", "test")
    inputs, targets, seen, in_ratings = [], [], [], []
    for i, s in enumerate(seqs):
        if mode == "valid":
            ctx, tgt = s[:-2], s[-2]
        else:
            ctx, tgt = s[:-1], s[-1]
        inputs.append(left_pad(ctx[-max_len:], max_len))
        targets.append(tgt)
        seen.append(ctx)
        if ratings is not None:
            r = ratings[i][: len(ctx)][-max_len:]
            in_ratings.append(left_pad(rating_bucket(r), max_len))
    out = np.stack(inputs), np.array(targets, dtype=np.int64), seen
    if ratings is not None:
        return *out, np.stack(in_ratings)
    return out
