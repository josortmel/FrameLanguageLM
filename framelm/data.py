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


def load_sequences(
    parquet: Path, vocab_path: Path, max_users: int | None = None
) -> tuple[list[np.ndarray], int]:
    """Devuelve (lista de secuencias por usuario en orden temporal, n_items)."""
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
    arr = con.execute(f"""
        SELECT s.userId, v.idx
        FROM read_parquet('{parquet}') s JOIN vocab v USING (tconst)
        {user_filter}
        ORDER BY s.userId, s.timestamp
    """).fetchnumpy()
    idx = np.ascontiguousarray(arr["idx"], dtype=np.int64)
    _, starts = np.unique(arr["userId"], return_index=True)
    bounds = np.append(starts, len(idx))
    seqs = [idx[bounds[i] : bounds[i + 1]] for i in range(len(starts))]
    # split leave-one-out necesita >=3 items (train>=1, valid, test)
    seqs = [s for s in seqs if len(s) >= 3]
    return seqs, len(mapping)


def left_pad(window: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    out[length - len(window) :] = window
    return out


class TrainDataset(Dataset):
    """Una muestra por usuario: ventana de los ultimos max_len+1 items de train."""

    def __init__(self, seqs: list[np.ndarray], max_len: int = MAX_LEN):
        self.max_len = max_len
        # train = s[:-2]; hace falta len>=2 ahi para tener al menos un par (x, y)
        self.train_seqs = [s[:-2] for s in seqs if len(s) >= 4]

    def __len__(self) -> int:
        return len(self.train_seqs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.train_seqs[i]
        window = s[-(self.max_len + 1) :]
        x = left_pad(window[:-1], self.max_len)
        y = left_pad(window[1:], self.max_len)
        return torch.from_numpy(x), torch.from_numpy(y)


def eval_batches(
    seqs: list[np.ndarray], mode: str, max_len: int = MAX_LEN
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(inputs (U, max_len), targets (U,), seen por usuario) para valid o test."""
    assert mode in ("valid", "test")
    inputs, targets, seen = [], [], []
    for s in seqs:
        if mode == "valid":
            ctx, tgt = s[:-2], s[-2]
        else:
            ctx, tgt = s[:-1], s[-1]
        inputs.append(left_pad(ctx[-max_len:], max_len))
        targets.append(tgt)
        seen.append(ctx)
    return np.stack(inputs), np.array(targets, dtype=np.int64), seen
