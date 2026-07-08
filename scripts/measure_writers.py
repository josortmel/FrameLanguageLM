"""F4: mide si los writers de series caen en el vocab de directores entrenado."""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

vocabs = json.loads((ROOT / "data/feature_vocabs.json").read_text(encoding="utf-8"))
director_vocab = set(vocabs["director"])

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW crew AS SELECT * FROM read_csv_auto(
        '{RAW / "title.crew.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
    CREATE VIEW names AS SELECT * FROM read_csv_auto(
        '{RAW / "name.basics.tsv.gz"}', delim='\t', quote='', nullstr='\\N');
""")

series = con.execute(
    """SELECT tconst FROM sqlite_scan(?, 'items') WHERE title_type != 'movie'""",
    [str(ROOT / "data/catalog.sqlite")],
).fetchall()
series_t = {r[0] for r in series}
print(f"series en catalogo: {len(series_t):,}", file=sys.stderr)

rows = con.execute("""
    WITH s AS (
        SELECT c.tconst, unnest(string_split(c.writers, ','))[1] AS dummy,
               string_split(c.writers, ',') AS ws
        FROM crew c
        WHERE c.writers IS NOT NULL
    )
    SELECT tconst, ws FROM s
""").fetchall()

# solo series del catalogo; primeros 3 writers; nconst->nombre
by_t = {t: ws[:3] for t, ws in rows if t in series_t}
all_nc = {nc for ws in by_t.values() for nc in ws}
print(f"series con writers: {len(by_t):,}; nconsts unicos: {len(all_nc):,}", file=sys.stderr)

name_rows = con.execute(
    """SELECT nconst, primaryName FROM names WHERE nconst IN
       (SELECT unnest(?::VARCHAR[]))""",
    [list(all_nc)],
).fetchall()
nc_name = dict(name_rows)

hit_series = 0
for t, ws in by_t.items():
    if any(nc_name.get(nc) in director_vocab for nc in ws):
        hit_series += 1

total = len(series_t)
print(json.dumps({
    "series_total": total,
    "series_con_writers": len(by_t),
    "series_con_writer_en_vocab_director": hit_series,
    "pct_sobre_total": round(100 * hit_series / total, 1),
    "pct_sobre_con_writers": round(100 * hit_series / max(len(by_t), 1), 1),
}, indent=2))
