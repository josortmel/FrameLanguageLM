"""Matching titulo (ES) + año -> tconst del catalogo.

Niveles: 1 exacto contra primary/original_title, 2 exacto contra titulos
españoles de IMDb akas (region ES), 3 fuzzy (rapidfuzz), 4 TMDB search.
Sin año (Netflix): el match exacto exige candidato unico o se desempata
por num_votes; el fuzzy sube el umbral. Nunca se descarta en silencio.
"""

import json
import os
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import httpx
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG = ROOT / "data" / "catalog.sqlite"
AKAS_ES = ROOT / "data" / "interim" / "akas_es.parquet"
AKAS_RAW = ROOT / "data" / "raw" / "title.akas.tsv.gz"

FUZZY_THRESHOLD_WITH_YEAR = 90
FUZZY_THRESHOLD_NO_YEAR = 94


def normalize(title: str) -> str:
    t = title.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _build_akas_es(catalog_tconsts: list[str]) -> None:
    AKAS_ES.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE vocab(tconst VARCHAR)")
    con.executemany("INSERT INTO vocab VALUES (?)", [(t,) for t in catalog_tconsts])
    con.execute(f"""
        COPY (
            SELECT a.titleId AS tconst, a.title
            FROM read_csv_auto('{AKAS_RAW}', delim='\t', quote='', nullstr='\\N') a
            JOIN vocab v ON v.tconst = a.titleId
            WHERE a.region = 'ES' AND a.title IS NOT NULL
        ) TO '{AKAS_ES}' (FORMAT PARQUET)
    """)


class Matcher:
    def __init__(self) -> None:
        con = sqlite3.connect(CATALOG)
        rows = con.execute(
            """SELECT tconst, title_type, primary_title, original_title,
                      start_year, num_votes, tmdb_id FROM items"""
        ).fetchall()
        con.close()

        self.by_tconst = {r[0]: r for r in rows}
        self.tmdb_to_tconst = {r[6]: r[0] for r in rows if r[6] is not None}
        self.exact: dict[str, list] = defaultdict(list)
        for r in rows:
            for title in {r[2], r[3]}:
                if title:
                    self.exact[normalize(title)].append(r)

        if not AKAS_ES.exists():
            _build_akas_es(list(self.by_tconst))
        akas = duckdb.connect().execute(
            f"SELECT tconst, title FROM read_parquet('{AKAS_ES}')"
        ).fetchall()
        self.akas: dict[str, list] = defaultdict(list)
        for tconst, title in akas:
            row = self.by_tconst.get(tconst)
            if row:
                self.akas[normalize(title)].append(row)

        # pool para fuzzy: norm_title -> candidatos (catalogo + akas)
        self.pool: dict[str, list] = defaultdict(list)
        for k, v in self.exact.items():
            self.pool[k].extend(v)
        for k, v in self.akas.items():
            self.pool[k].extend(v)
        self.pool_keys = list(self.pool)

        token = None
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("TMDB_READ_TOKEN="):
                    token = line.split("=", 1)[1].strip()
        self.tmdb_token = token or os.environ.get("TMDB_READ_TOKEN")

    def _pick(self, cands: list, year: int | None, series_hint: bool | None):
        if series_hint is True:
            cands = [c for c in cands if c[1] != "movie"] or cands
        if year is not None:
            cands = [c for c in cands if c[4] and abs(c[4] - year) <= 1]
            if not cands:
                return None
        if len(cands) == 1:
            return cands[0]
        if cands and year is None:
            return max(cands, key=lambda c: c[5] or 0)  # desempate: num_votes
        return cands[0] if cands else None

    def _tmdb(self, title: str, year: int | None, series_hint: bool | None):
        if not self.tmdb_token:
            return None
        headers = {"Authorization": f"Bearer {self.tmdb_token}"}
        kinds = ["tv", "movie"] if series_hint else ["movie", "tv"]
        for kind in kinds:
            params = {"query": title, "language": "es-ES"}
            if year is not None:
                params["year" if kind == "movie" else "first_air_date_year"] = year
            try:
                r = httpx.get(
                    f"https://api.themoviedb.org/3/search/{kind}",
                    params=params, headers=headers, timeout=15,
                )
                results = r.json().get("results") or []
            except Exception:
                return None
            for res in results[:3]:
                tconst = self.tmdb_to_tconst.get(res["id"])
                if tconst:
                    return self.by_tconst[tconst]
        return None

    def match(
        self, title: str, year: int | None = None, series_hint: bool | None = None
    ) -> tuple[str | None, int]:
        """Devuelve (tconst, nivel 1-4) o (None, 0)."""
        norm = normalize(title)

        for level, table in ((1, self.exact), (2, self.akas)):
            row = self._pick(table.get(norm, []), year, series_hint)
            if row:
                return row[0], level

        threshold = (
            FUZZY_THRESHOLD_WITH_YEAR if year is not None
            else FUZZY_THRESHOLD_NO_YEAR
        )
        hit = process.extractOne(
            norm, self.pool_keys, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if hit:
            row = self._pick(self.pool[hit[0]], year, series_hint)
            if row:
                return row[0], 3

        row = self._tmdb(title, year, series_hint)
        if row:
            return row[0], 4

        # fallback: titulos compuestos de Netflix ("Serie: Nombre de temporada")
        # reintenta con el segmento base solo para series
        if series_hint and ": " in title:
            base = title.split(": ", 1)[0].strip()
            if len(base) >= 3:
                return self.match(base, year, series_hint=True)
        return None, 0
