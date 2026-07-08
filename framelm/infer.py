"""Inferencia CPU dual sobre artefactos ONNX.

Dos poblaciones, dos modelos, scores nunca mezclados:
- warm (54k entrenables, cine sobre todo) -> checkpoint feat, calidad plena.
- cold (46k restantes, series/docs/oscuras) -> checkpoint iddrop (unico
  entrenado para vectores solo-torre), con calibracion: coseno + suelo de
  ficha + prior de popularidad.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import onnxruntime as ort

MAX_LEN = 200
COLD_ALPHA = 0.5  # peso del prior de popularidad en el ranking frio


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, int((scores > -np.inf).sum()))
    top = np.argpartition(scores, -k)[-k:]
    return top[np.argsort(scores[top])[::-1]]


def _z(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = x[valid]
    mu, sd = float(v.mean()), float(v.std() or 1.0)
    return (x - mu) / sd


class FrameLM:
    def __init__(self, artifacts_dir: str | Path = "data/artifacts",
                 catalog: str | Path = "data/catalog.sqlite",
                 data_dir: str | Path = "data"):
        a = Path(artifacts_dir)
        d = Path(data_dir)

        # --- path caliente (feat) ---
        self.warm_session = _session(a / "model_fp32.onnx")
        self.warm_matrix = np.load(a / "item_embeddings.npy")
        wn = np.linalg.norm(self.warm_matrix, axis=1, keepdims=True)
        wn[wn == 0] = 1.0
        self._warm_unit = self.warm_matrix / wn
        warm_map = json.loads((d / "vocab_map.json").read_text("utf-8"))["tconst_to_idx"]
        self.n_warm = len(warm_map)

        # --- path frio (iddrop, catalogo completo) ---
        self.cold_session = _session(a / "model_full_fp32.onnx")
        self.full_matrix = np.load(a / "item_embeddings_full.npy")
        fn = np.linalg.norm(self.full_matrix, axis=1, keepdims=True)
        fn[fn == 0] = 1.0
        self._full_unit = self.full_matrix / fn

        full_map = json.loads((d / "full_vocab_map.json").read_text("utf-8"))["tconst_to_idx"]
        self.tconst_to_idx = full_map           # warm ids coinciden (filas 1..54053)
        self.idx_to_tconst = {i: t for t, i in full_map.items()}

        aux = np.load(a / "full_aux.npz")
        self.aux = {k: aux[k] for k in aux.files}
        # suelo de ficha F2: frio recomendable solo con director O >=2 cast
        self._cold_ok = self.aux["is_cold"] & (
            self.aux["director_known"] | (self.aux["cast_known_n"] >= 2)
        )
        self._log_votes = np.log10(1.0 + self.aux["num_votes"])

        con = duckdb.connect(str(catalog), read_only=True)
        rows = con.execute(
            "SELECT tconst, primary_title, start_year, title_type FROM items"
        ).fetchall()
        con.close()
        self.titles = {t: (title, year, tt) for t, title, year, tt in rows}

    # ---------- interno ----------

    def _hidden(self, session: ort.InferenceSession, seq: list[int],
                max_idx: int | None = None) -> np.ndarray:
        window = [i for i in seq if i > 0 and (max_idx is None or i <= max_idx)]
        window = window[-MAX_LEN:]
        x = np.zeros((1, MAX_LEN), dtype=np.int64)
        if window:
            x[0, MAX_LEN - len(window):] = window
        return session.run(None, {"seq": x})[0][0]

    def _describe(self, idx: int) -> dict:
        tconst = self.idx_to_tconst.get(idx, "?")
        title, year, tt = self.titles.get(tconst, ("?", None, "?"))
        return {"tconst": tconst, "title": title, "year": year, "type": tt}

    def _rank_warm(self, seq: list[int], k: int, allowed: np.ndarray) -> list[dict]:
        scores = self.warm_matrix @ self._hidden(
            self.warm_session, seq, max_idx=self.n_warm
        )
        scores[0] = -np.inf
        seen = [i for i in set(seq) if 0 < i < len(scores)]
        scores[seen] = -np.inf
        scores[~allowed[: len(scores)]] = -np.inf
        return [self._describe(int(i)) | {"score": float(scores[i]), "pop": "warm"}
                for i in _top(scores, k)]

    def _rank_cold(self, seq: list[int], k: int, allowed: np.ndarray) -> list[dict]:
        """F1 coseno + F2 suelo de ficha + F3 prior de popularidad, solo frios."""
        h = self._hidden(self.cold_session, seq)
        h = h / (np.linalg.norm(h) or 1.0)
        cos = self._full_unit @ h
        valid = allowed & self._cold_ok
        if not valid.any():
            return []
        score = _z(cos, valid) + COLD_ALPHA * _z(self._log_votes, valid)
        score[~valid] = -np.inf
        seen_t = {self.idx_to_tconst.get(i) for i in seq}
        for t in seen_t:
            j = self.tconst_to_idx.get(t)
            if j is not None:
                score[j] = -np.inf
        return [self._describe(int(i)) | {"score": float(score[i]), "pop": "cold"}
                for i in _top(score, k)]

    # ---------- API ----------

    def gaps_movies(self, seq: list[int], k: int = 50) -> list[dict]:
        """Cine del vocab entrenable, modelo feat — calidad plena."""
        allowed = self.aux["is_movie"] & ~self.aux["is_cold"]
        return self._rank_warm(seq, k, allowed)

    def gaps_series(self, seq: list[int], k: int = 50) -> list[dict]:
        """Series: bloque warm (feat, las pocas series entrenadas) +
        bloque cold (iddrop calibrado). Sin mezclar scores."""
        warm = self._rank_warm(seq, k, self.aux["is_series"] & ~self.aux["is_cold"])
        cold = self._rank_cold(seq, k, self.aux["is_series"])
        return {"warm": warm, "cold": cold}

    def gaps_docs(self, seq: list[int], k: int = 50) -> dict:
        warm = self._rank_warm(seq, k, self.aux["is_doc"] & ~self.aux["is_cold"])
        cold = self._rank_cold(seq, k, self.aux["is_doc"])
        return {"warm": warm, "cold": cold}

    # compat: ranking generico warm (comportamiento fase 4)
    def gaps(self, seq: list[int], k: int = 50, **_) -> list[dict]:
        return self._rank_warm(seq, k, ~self.aux["is_cold"])

    def worth(self, tconst: str, seq: list[int]) -> dict:
        idx = self.tconst_to_idx.get(tconst)
        if idx is None:
            raise KeyError(f"{tconst} fuera del catalogo")
        cold = bool(self.aux["is_cold"][idx])
        if cold:
            h = self._hidden(self.cold_session, seq)
            h = h / (np.linalg.norm(h) or 1.0)
            scores = self._full_unit @ h
            pool = self._cold_ok.copy()
        else:
            scores = np.full(len(self.full_matrix), -np.inf, dtype=np.float32)
            w = self.warm_matrix @ self._hidden(
                self.warm_session, seq, max_idx=self.n_warm
            )
            scores[: len(w)] = w
            pool = ~self.aux["is_cold"]
            pool[0] = False
        seen = {self.tconst_to_idx.get(self.idx_to_tconst.get(i)) for i in seq}
        pool[[i for i in seen if i is not None]] = False
        pct = float((scores[pool] < scores[idx]).mean() * 100.0)
        return self._describe(idx) | {
            "score": float(scores[idx]), "percentile": pct,
            "population": "cold" if cold else "warm",
        }

    def neighbors(self, tconst: str, k: int = 10) -> list[dict]:
        idx = self.tconst_to_idx.get(tconst)
        if idx is None:
            raise KeyError(f"{tconst} fuera del catalogo")
        if self.aux["is_cold"][idx]:
            unit, floor = self._full_unit, self._cold_ok | ~self.aux["is_cold"]
        else:
            unit = self._warm_unit
            floor = np.ones(len(unit), dtype=bool)
        sims = unit @ unit[min(idx, len(unit) - 1)]
        sims[~floor[: len(sims)]] = -np.inf
        sims[0] = -np.inf
        if idx < len(sims):
            sims[idx] = -np.inf
        return [self._describe(int(j)) | {"sim": float(sims[j])}
                for j in _top(sims, k)]
