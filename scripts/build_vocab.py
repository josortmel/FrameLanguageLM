"""Fase 1.2: vocabulario de 100k titulos con features IMDb.

Uso: uv run python scripts/build_vocab.py
Entrada: data/raw/ (IMDb TSVs + ml-32m/links.csv)
Salida: data/interim/vocab.parquet
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
VOCAB_SIZE = 100_000

INTERIM.mkdir(parents=True, exist_ok=True)
out = INTERIM / "vocab.parquet"

con = duckdb.connect()

con.execute(f"""
    CREATE VIEW basics AS SELECT * FROM read_csv_auto(
        '{RAW / "title.basics.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW tratings AS SELECT * FROM read_csv_auto(
        '{RAW / "title.ratings.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW crew AS SELECT * FROM read_csv_auto(
        '{RAW / "title.crew.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW principals AS SELECT * FROM read_csv_auto(
        '{RAW / "title.principals.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW names AS SELECT * FROM read_csv_auto(
        '{RAW / "name.basics.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW links AS SELECT * FROM read_csv_auto('{RAW / "ml-32m" / "links.csv"}');
""")

print("seleccionando vocabulario...")
con.execute(f"""
    CREATE TABLE vocab AS
    SELECT b.tconst, b.titleType, b.primaryTitle, b.originalTitle, b.startYear,
           b.runtimeMinutes, b.genres, r.averageRating, r.numVotes
    FROM basics b JOIN tratings r USING (tconst)
    WHERE b.titleType IN ('movie', 'tvSeries', 'tvMiniSeries')
      AND b.isAdult = 0
    ORDER BY r.numVotes DESC
    LIMIT {VOCAB_SIZE}
""")

print("agregando directores...")
con.execute("""
    CREATE TABLE directors AS
    SELECT c.tconst, string_agg(n.primaryName, '|') AS directors
    FROM (
        SELECT tconst, unnest(string_split(directors, ',')) AS nconst
        FROM crew
        WHERE directors IS NOT NULL
          AND tconst IN (SELECT tconst FROM vocab)
    ) c JOIN names n USING (nconst)
    GROUP BY c.tconst
""")

print("agregando cast top-6 (title.principals es grande, paciencia)...")
con.execute("""
    CREATE TABLE cast6 AS
    SELECT q.tconst, string_agg(n.primaryName, '|' ORDER BY q.ordering) AS "cast"
    FROM (
        SELECT tconst, ordering, nconst
        FROM principals
        WHERE category IN ('actor', 'actress')
          AND tconst IN (SELECT tconst FROM vocab)
        QUALIFY row_number() OVER (PARTITION BY tconst ORDER BY ordering) <= 6
    ) q JOIN names n USING (nconst)
    GROUP BY q.tconst
""")

# printf en vez de lpad: los tconst recientes tienen 8 digitos y lpad truncaria
con.execute("""
    CREATE TABLE linkmap AS
    SELECT printf('tt%07d', CAST(imdbId AS BIGINT)) AS tconst, max(tmdbId) AS tmdb_id_links
    FROM links WHERE tmdbId IS NOT NULL
    GROUP BY 1
""")

con.execute(f"""
    COPY (
        SELECT v.*, d.directors, c."cast", l.tmdb_id_links
        FROM vocab v
        LEFT JOIN directors d USING (tconst)
        LEFT JOIN cast6 c USING (tconst)
        LEFT JOIN linkmap l USING (tconst)
        ORDER BY v.numVotes DESC
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
""")

n, d, c, g, t = con.execute(f"""
    SELECT count(*),
           100.0 * count(directors) / count(*),
           100.0 * count("cast") / count(*),
           100.0 * count(genres) / count(*),
           count(tmdb_id_links)
    FROM read_parquet('{out.as_posix()}')
""").fetchone()

print(f"\nvocab.parquet: {n:,} filas")
print(f"  cobertura directores: {d:.1f}%")
print(f"  cobertura cast:       {c:.1f}%")
print(f"  cobertura genres:     {g:.1f}%")
print(f"  con tmdbId de links:  {t:,} ({100 * t / n:.1f}%)")
