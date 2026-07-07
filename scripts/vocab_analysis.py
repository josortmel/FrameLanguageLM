"""Fase 1.1: analisis del corte de vocabulario.

Para cada corte candidato (top-N titulos por numVotes de IMDb, tipos
movie/tvSeries/tvMiniSeries) calcula:
  - numVotes minimo del titulo que entra (el "mas pequeño")
  - % de ratings de ML-32M retenidos al filtrar al vocabulario
  - composicion del corte (peliculas vs series)

Uso: uv run python scripts/vocab_analysis.py
"""

import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CUTS = [25_000, 50_000, 75_000, 100_000, 150_000]

# duckdb no lee dentro de zips: extraer los csv de ML una vez
ml_dir = RAW / "ml-32m"
if not ml_dir.exists():
    with zipfile.ZipFile(RAW / "ml-32m.zip") as z:
        z.extractall(RAW)

con = duckdb.connect()

con.execute(f"""
    CREATE VIEW basics AS SELECT * FROM read_csv_auto(
        '{RAW / "title.basics.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW tratings AS SELECT * FROM read_csv_auto(
        '{RAW / "title.ratings.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW links AS SELECT * FROM read_csv_auto('{ml_dir / "links.csv"}');
    CREATE VIEW ml AS SELECT * FROM read_csv_auto('{ml_dir / "ratings.csv"}');
""")

con.execute("""
    CREATE TABLE candidates AS
    SELECT b.tconst, b.titleType, b.primaryTitle, b.startYear, r.numVotes
    FROM basics b JOIN tratings r USING (tconst)
    WHERE b.titleType IN ('movie', 'tvSeries', 'tvMiniSeries')
      AND b.isAdult = 0
    ORDER BY r.numVotes DESC
""")

total_candidates = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
total_ml = con.execute("SELECT count(*) FROM ml").fetchone()[0]

# ratings de ML mapeados a tconst (links.imdbId es el tconst sin prefijo 'tt')
con.execute("""
    CREATE TABLE ml_tconst AS
    SELECT 'tt' || lpad(CAST(l.imdbId AS VARCHAR), 7, '0') AS tconst,
           count(*) AS n_ratings
    FROM ml m JOIN links l USING (movieId)
    GROUP BY 1
""")

print(f"candidatos totales (movie/tvSeries/tvMini con rating): {total_candidates:,}")
print(f"ratings ML-32M totales: {total_ml:,}\n")
print(f"{'corte':>8} | {'numVotes min':>12} | {'% ratings ML':>12} | {'pelis':>7} | {'series':>7}")
print("-" * 62)

for cut in CUTS:
    row = con.execute(f"""
        WITH vocab AS (SELECT * FROM candidates LIMIT {cut})
        SELECT
            (SELECT min(numVotes) FROM vocab),
            (SELECT coalesce(sum(n_ratings), 0) FROM ml_tconst
             WHERE tconst IN (SELECT tconst FROM vocab)),
            (SELECT count(*) FROM vocab WHERE titleType = 'movie'),
            (SELECT count(*) FROM vocab WHERE titleType != 'movie')
    """).fetchone()
    min_votes, ml_kept, n_movies, n_series = row
    print(f"{cut:>8,} | {min_votes:>12,} | {100 * ml_kept / total_ml:>11.1f}% | {n_movies:>7,} | {n_series:>7,}")

print("\nejemplos del titulo 'mas pequeño' en cada corte:")
for cut in CUTS:
    title, year, votes = con.execute(
        f"SELECT primaryTitle, startYear, numVotes FROM candidates LIMIT 1 OFFSET {cut - 1}"
    ).fetchone()
    print(f"  top {cut:>7,}: {title} ({year}) — {votes:,} votos")
