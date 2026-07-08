"""Inferencia CPU sobre artefactos ONNX: gaps, worth, neighbors.

Scoring fuera del grafo: hidden (d,) @ item_matrix.T con numpy brute-force.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import onnxruntime as ort

MAX_LEN = 200


class FrameLM:
    def __init__(self, artifacts_dir: str | Path = "data/artifacts",
                 catalog: str | Path = "data/catalog.sqlite",
                 vocab: str | Path = "data/vocab_map.json"):
        artifacts_dir = Path(artifacts_dir)
        # Si existe el artefacto de catalogo completo (cold-start incluido),
        # se usa; si no, cae al vocabulario entrenable de 54k.
        full_onnx = artifacts_dir / "model_full_fp32.onnx"
        full_vocab = artifacts_dir.parent / "full_vocab_map.json"
        self.full = full_onnx.exists() and full_vocab.exists()
        onnx_path = full_onnx if self.full else artifacts_dir / "model_fp32.onnx"
        matrix_path = artifacts_dir / (
            "item_embeddings_full.npy" if self.full else "item_embeddings.npy"
        )
        if self.full:
            vocab = full_vocab
        # produccion = fp32: el int8 dinamico de ORT degrada el ranking ~70%
        # en este modelo y no ahorra tamano relevante (ver export_onnx.py).
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.model_kind = "fp32-full" if self.full else "fp32"

        self.matrix = np.load(matrix_path)  # (n+1, d)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._unit = self.matrix / norms

        mapping = json.loads(Path(vocab).read_text(encoding="utf-8"))["tconst_to_idx"]
        self.tconst_to_idx: dict[str, int] = mapping
        self.idx_to_tconst = {i: t for t, i in mapping.items()}

        con = duckdb.connect(str(catalog), read_only=True)
        rows = con.execute(
            "SELECT tconst, primary_title, start_year FROM items"
        ).fetchall()
        con.close()
        self.titles = {t: (title, year) for t, title, year in rows}

    # ---------- interno ----------

    def _hidden(self, seq_indices: list[int]) -> np.ndarray:
        window = [i for i in seq_indices if i > 0][-MAX_LEN:]
        x = np.zeros((1, MAX_LEN), dtype=np.int64)
        if window:
            x[0, MAX_LEN - len(window):] = window
        return self.session.run(None, {"seq": x})[0][0]  # (d,)

    def _scores(self, seq_indices: list[int]) -> np.ndarray:
        s = self.matrix @ self._hidden(seq_indices)  # (n+1,)
        s[0] = -np.inf
        return s

    def _describe(self, idx: int) -> tuple[str, str, int | None]:
        tconst = self.idx_to_tconst.get(idx, "?")
        title, year = self.titles.get(tconst, ("?", None))
        return tconst, title, year

    # ---------- API ----------

    def gaps(self, seq_indices: list[int], k: int = 50,
             exclude: set[int] | None = None,
             mask: "np.ndarray | None" = None) -> list[dict]:
        """mask: bool (n+1,) — True = candidato permitido (busqueda fina)."""
        scores = self._scores(seq_indices)
        seen = set(seq_indices) if exclude is None else set(exclude)
        seen.discard(0)
        scores[list(seen)] = -np.inf
        if mask is not None:
            scores[~mask] = -np.inf
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        out = []
        for idx in top:
            tconst, title, year = self._describe(int(idx))
            out.append(
                {"tconst": tconst, "title": title, "year": year,
                 "score": float(scores[idx])}
            )
        return out

    def worth(self, tconst: str, seq_indices: list[int]) -> dict:
        idx = self.tconst_to_idx.get(tconst)
        if idx is None:
            raise KeyError(f"{tconst} fuera del vocabulario entrenable")
        scores = self._scores(seq_indices)
        seen = set(seq_indices)
        seen.discard(0)
        unseen = np.ones(len(scores), dtype=bool)
        unseen[list(seen)] = False
        unseen[0] = False
        pool = scores[unseen]
        pct = float((pool < scores[idx]).mean() * 100.0)
        _, title, year = self._describe(idx)
        return {"tconst": tconst, "title": title, "year": year,
                "score": float(scores[idx]), "percentile": pct}

    def neighbors(self, tconst: str, k: int = 10) -> list[dict]:
        idx = self.tconst_to_idx.get(tconst)
        if idx is None:
            raise KeyError(f"{tconst} fuera del vocabulario entrenable")
        sims = self._unit @ self._unit[idx]
        sims[0] = -np.inf
        sims[idx] = -np.inf
        top = np.argpartition(sims, -k)[-k:]
        top = top[np.argsort(sims[top])[::-1]]
        out = []
        for j in top:
            t, title, year = self._describe(int(j))
            out.append({"tconst": t, "title": title, "year": year,
                        "sim": float(sims[j])})
        return out
