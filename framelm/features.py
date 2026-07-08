"""Features composicionales por item desde catalog.sqlite.

Alineadas con vocab_map.json: fila i = item con indice i (0 = <pad>, features
nulas). En cada vocabulario de feature, 0 = desconocido/null.

Uso: uv run python -m framelm.features  (genera data/features.npz +
data/feature_vocabs.json)
"""

import json
import math
from pathlib import Path

import duckdb
import numpy as np

N_DIRECTORS = 3
N_CAST = 4
N_GENRES = 3

BUDGET_EDGES = [1e6, 5e6, 20e6, 50e6, 150e6]  # buckets: <1M,1-5,5-20,20-50,50-150,>150


def budget_bucket(budget) -> int:
    if budget is None or budget <= 0:
        return 0
    for i, edge in enumerate(BUDGET_EDGES):
        if budget < edge:
            return i + 1
    return len(BUDGET_EDGES) + 1


def decade_bucket(year) -> int:
    if year is None or year < 1920:
        return 0
    return min((int(year) - 1920) // 10 + 1, 12)  # 1920s..2030s


def _split(value: str | None, sep: str, cap: int) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(sep) if v.strip()][:cap]


class _Vocab:
    def __init__(self):
        self.map: dict[str, int] = {}

    def get(self, key: str) -> int:
        if key not in self.map:
            self.map[key] = len(self.map) + 1  # 0 reservado a desconocido
        return self.map[key]


def build_features(
    catalog: Path = Path("data/catalog.sqlite"),
    vocab_path: Path = Path("data/vocab_map.json"),
    out_npz: Path = Path("data/features.npz"),
    out_vocabs: Path = Path("data/feature_vocabs.json"),
) -> None:
    mapping = json.loads(vocab_path.read_text(encoding="utf-8"))["tconst_to_idx"]
    n = len(mapping)

    con = duckdb.connect(str(catalog), read_only=True)
    rows = con.execute(
        """SELECT tconst, directors, "cast", genres, countries,
           original_language, start_year, budget FROM items"""
    ).fetchall()

    directors = _Vocab()
    actors = _Vocab()
    genres_v = _Vocab()
    countries_v = _Vocab()
    langs = _Vocab()

    dir_idx = np.zeros((n + 1, N_DIRECTORS), dtype=np.int64)
    cast_idx = np.zeros((n + 1, N_CAST), dtype=np.int64)
    gen_idx = np.zeros((n + 1, N_GENRES), dtype=np.int64)
    country = np.zeros(n + 1, dtype=np.int64)
    language = np.zeros(n + 1, dtype=np.int64)
    decade = np.zeros(n + 1, dtype=np.int64)
    budget = np.zeros(n + 1, dtype=np.int64)

    found = 0
    for tconst, dirs, cast, gens, ctry, lang, year, budg in rows:
        i = mapping.get(tconst)
        if i is None:
            continue  # item del catalogo sin señal colaborativa: cold-start futuro
        found += 1
        for j, name in enumerate(_split(dirs, "|", N_DIRECTORS)):
            dir_idx[i, j] = directors.get(name)
        for j, name in enumerate(_split(cast, "|", N_CAST)):
            cast_idx[i, j] = actors.get(name)
        for j, g in enumerate(_split(gens, ",", N_GENRES)):
            gen_idx[i, j] = genres_v.get(g)
        c = _split(ctry, "|", 1)
        if c:
            country[i] = countries_v.get(c[0])
        if lang:
            language[i] = langs.get(lang)
        decade[i] = decade_bucket(year)
        budget[i] = budget_bucket(budg)

    np.savez_compressed(
        out_npz,
        director=dir_idx, cast=cast_idx, genre=gen_idx,
        country=country, language=language, decade=decade, budget=budget,
    )
    out_vocabs.write_text(
        json.dumps(
            {
                "director": directors.map, "cast": actors.map,
                "genre": genres_v.map, "country": countries_v.map,
                "language": langs.map,
                "n_decade": 13, "n_budget": len(BUDGET_EDGES) + 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cov_dir = (dir_idx[1:, 0] > 0).mean()
    cov_country = (country[1:] > 0).mean()
    print(
        f"items con features: {found:,}/{n:,} | "
        f"cobertura director {cov_dir:.1%}, pais {cov_country:.1%} | "
        f"vocabs: {len(directors.map):,} directores, {len(actors.map):,} actores, "
        f"{len(genres_v.map)} generos, {len(countries_v.map)} paises, "
        f"{len(langs.map)} idiomas"
    )


if __name__ == "__main__":
    build_features()
