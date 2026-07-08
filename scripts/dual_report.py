"""Reporte de servido dual para juicio humano (secuencia real de Pepe)."""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framelm.infer import FrameLM  # noqa: E402


def show(tag: str, items: list[dict]) -> None:
    print(f"\n[{tag}]")
    for it in items[:10]:
        cold = "*" if it.get("pop") == "cold" else " "
        print(f" {cold} {it['score']:6.2f}  {it['title']} ({it['year']}) [{it['type']}]")


def main() -> None:
    lm = FrameLM(ROOT / "data/artifacts", ROOT / "data/catalog.sqlite", ROOT / "data")

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT tconst, indice FROM '{ROOT / 'data/user/pepe_sequence.parquet'}' "
        "ORDER BY timestamp"
    ).fetchall()
    con.close()

    seq_warm = [i for _, i in rows if i and i > 0]
    full_map = lm.tconst_to_idx
    seq_full = [full_map[t] for t, _ in rows if t in full_map]
    n_unk_mapped = len(seq_full) - len(seq_warm)
    print(f"eventos: warm-input={len(seq_warm)}, full-input={len(seq_full)} "
          f"(+{n_unk_mapped} unk remapeados a filas frias)")

    print("\n================ PELICULAS (path feat, calidad plena) ================")
    show("MOVIES top-10", lm.gaps_movies(seq_warm, 10))

    print("\n================ SERIES (path iddrop calibrado) ================")
    for label, seq in (("input=warm(456)", seq_warm), ("input=full(705)", seq_full)):
        s = lm.gaps_series(seq, 10)
        show(f"SERIES warm-block {label}", s["warm"])
        show(f"SERIES cold-block {label}", s["cold"])

    print("\n================ DOCUMENTALES ================")
    d = lm.gaps_docs(seq_warm, 10)
    show("DOCS warm-block", d["warm"])
    show("DOCS cold-block", d["cold"])

    print("\n================ VECINOS (F2+F4 aplicados) ================")
    for name, t in (("Breaking Bad", "tt0903747"), ("Dark", "tt5753856")):
        print(f"\n{name}:")
        for n in lm.neighbors(t, 10):
            print(f"  {n['sim']:.3f}  {n['title']} ({n['year']}) [{n['type']}]")


if __name__ == "__main__":
    main()
