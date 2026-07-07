"""Fase 1.4: ensambla el catalogo final y las secuencias de entrenamiento.

Uso: uv run python scripts/build_catalog.py
Entrada: data/interim/vocab.parquet + data/interim/tmdb.jsonl + data/raw/ml-32m/
Salida: data/catalog.sqlite + data/sequences.parquet
"""

import json
import sqlite3
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
MIN_TMDB_COVERAGE = 0.80

vocab = duckdb.sql(
    f"SELECT * FROM read_parquet('{(INTERIM / 'vocab.parquet').as_posix()}') ORDER BY numVotes DESC"
).fetchall()
cols = [
    "tconst", "titleType", "primaryTitle", "originalTitle", "startYear",
    "runtimeMinutes", "genres", "averageRating", "numVotes",
    "directors", "cast", "tmdb_id_links",
]

tmdb: dict[str, dict] = {}
tmdb_path = INTERIM / "tmdb.jsonl"
if tmdb_path.exists():
    with tmdb_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "error" not in rec:
                    tmdb[rec["tconst"]] = rec

coverage = len([1 for row in vocab if row[0] in tmdb]) / len(vocab)
if coverage < MIN_TMDB_COVERAGE:
    sys.exit(
        f"cobertura TMDB insuficiente: {coverage:.1%} del vocabulario "
        f"(minimo {MIN_TMDB_COVERAGE:.0%}). Lanza scripts/fetch_tmdb.py hasta completar."
    )

db_path = ROOT / "data" / "catalog.sqlite"
db_path.unlink(missing_ok=True)
db = sqlite3.connect(db_path)
db.execute("""
    CREATE TABLE items (
        tconst TEXT PRIMARY KEY,
        title_type TEXT NOT NULL,
        primary_title TEXT NOT NULL,
        original_title TEXT,
        start_year INTEGER,
        runtime_minutes INTEGER,
        genres TEXT,
        imdb_rating REAL,
        num_votes INTEGER NOT NULL,
        directors TEXT,
        "cast" TEXT,
        tmdb_id INTEGER,
        media_type TEXT,
        original_language TEXT,
        countries TEXT,
        budget INTEGER,
        keywords TEXT,
        poster_path TEXT,
        popularity REAL
    )
""")

rows = []
for row in vocab:
    r = dict(zip(cols, row))
    t = tmdb.get(r["tconst"], {})
    rows.append((
        r["tconst"], r["titleType"], r["primaryTitle"], r["originalTitle"],
        r["startYear"], r["runtimeMinutes"], r["genres"], r["averageRating"],
        r["numVotes"], r["directors"], r["cast"],
        t.get("tmdb_id"), t.get("media_type"), t.get("original_language"),
        "|".join(t.get("production_countries") or []) or None,
        t.get("budget") or None,
        "|".join(t.get("keywords") or []) or None,
        t.get("poster_path"), t.get("popularity"),
    ))
db.executemany(f"INSERT INTO items VALUES ({','.join('?' * 19)})", rows)
db.execute("CREATE INDEX idx_items_title ON items (primary_title)")
db.commit()
db.close()
print(f"catalog.sqlite: {len(rows):,} items (cobertura TMDB {coverage:.1%})")

seq_out = ROOT / "data" / "sequences.parquet"
duckdb.sql(f"""
    COPY (
        SELECT m.userId, v.tconst, m.rating, m.timestamp
        FROM read_csv_auto('{(RAW / "ml-32m" / "ratings.csv").as_posix()}') m
        JOIN read_csv_auto('{(RAW / "ml-32m" / "links.csv").as_posix()}') l USING (movieId)
        JOIN read_parquet('{(INTERIM / "vocab.parquet").as_posix()}') v
          ON v.tconst = printf('tt%07d', CAST(l.imdbId AS BIGINT))
        ORDER BY m.userId, m.timestamp
    ) TO '{seq_out.as_posix()}' (FORMAT PARQUET)
""")
n, users = duckdb.sql(
    f"SELECT count(*), count(DISTINCT userId) FROM read_parquet('{seq_out.as_posix()}')"
).fetchone()
print(f"sequences.parquet: {n:,} interacciones de {users:,} usuarios")
