"""Fase 5: importa datos de usuario (FilmAffinity/Netflix) a secuencia.

Uso:
  uv run python scripts/import_user.py --filmaffinity "ruta.zip" \
      --netflix "ViewingActivity.csv" --profile "Jose Antonio" \
      [--out data/user] [--name pepe]

Salida: <out>/<name>_sequence.parquet + <out>/import_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame_language_lm.importers import parse_filmaffinity, parse_netflix
from frame_language_lm.importers.matching import Matcher

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filmaffinity")
    p.add_argument("--netflix")
    p.add_argument("--profile")
    p.add_argument("--out", default="data/user")
    p.add_argument("--name", default="pepe")
    args = p.parse_args()

    events = []
    if args.filmaffinity:
        events += parse_filmaffinity(Path(args.filmaffinity))
    if args.netflix:
        if not args.profile:
            sys.exit("--netflix requiere --profile")
        events += parse_netflix(Path(args.netflix), args.profile)
    if not events:
        sys.exit("nada que importar")

    vocab = json.loads((ROOT / "data" / "vocab_map.json").read_text("utf-8"))
    tconst_to_idx = vocab["tconst_to_idx"]

    matcher = Matcher()
    rows, unmatched, out_of_vocab = [], [], []
    by_level = {1: 0, 2: 0, 3: 0, 4: 0}
    by_source_total, by_source_matched = {}, {}

    for ev in events:
        src = ev["source"]
        by_source_total[src] = by_source_total.get(src, 0) + 1
        tconst, level = matcher.match(ev["title"], ev["year"], ev["series_hint"])
        if tconst is None:
            unmatched.append(f"{ev['title']} ({ev['year']}) [{src}]")
            continue
        by_level[level] += 1
        by_source_matched[src] = by_source_matched.get(src, 0) + 1
        idx = tconst_to_idx.get(tconst)
        if idx is None:
            out_of_vocab.append(f"{ev['title']} -> {tconst}")
            idx = -1  # <unk>: fuera del vocab entrenable
        rows.append(
            (tconst, idx, ev["rating"], ev["timestamp"], src, ev["title"])
        )

    rows.sort(key=lambda r: (r[3] is None, r[3]))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"{args.name}_sequence.parquet"
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE seq (tconst VARCHAR, indice BIGINT, rating DOUBLE,
                          timestamp TIMESTAMP, fuente VARCHAR, titulo VARCHAR)
    """)
    con.executemany("INSERT INTO seq VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY seq TO '{parquet}' (FORMAT PARQUET)")

    report = {
        "total_eventos": len(events),
        "matcheados": len(rows),
        "matching_por_nivel": by_level,
        "por_fuente": {
            s: {
                "total": by_source_total[s],
                "matcheados": by_source_matched.get(s, 0),
                "pct": round(
                    100 * by_source_matched.get(s, 0) / by_source_total[s], 1
                ),
            }
            for s in by_source_total
        },
        "no_matcheados": unmatched,
        "fuera_de_vocab": sorted(set(out_of_vocab)),
        "secuencia_final": len(rows),
        "en_vocab_entrenable": sum(1 for r in rows if r[1] > 0),
    }
    (out_dir / "import_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), "utf-8"
    )

    print(f"secuencia: {parquet} ({len(rows)} eventos)")
    print(json.dumps({k: v for k, v in report.items() if k != "no_matcheados"},
                     indent=2, ensure_ascii=False))
    print(f"no matcheados: {len(unmatched)}")
    for t in unmatched:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
