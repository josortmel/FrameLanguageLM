"""Smoke del servido dual: worth warm/cold + vecinos warm."""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frame_language_lm.infer import FrameLM  # noqa: E402

lm = FrameLM(ROOT / "data/artifacts", ROOT / "data/catalog.sqlite", ROOT / "data")
rows = duckdb.sql(
    f"SELECT tconst, indice FROM '{ROOT / 'data/user/pepe_sequence.parquet'}' "
    "ORDER BY timestamp"
).fetchall()
seq = [i for _, i in rows if i and i > 0]

print("worth Stalker (warm):", lm.worth("tt0079944", seq))
print("worth Silo (cold):", lm.worth("tt14688458", seq))
print("vecinos Persona (warm):", [n["title"] for n in lm.neighbors("tt0060827", 5)])
assert lm.worth("tt0079944", seq)["population"] == "warm"
assert lm.worth("tt14688458", seq)["population"] == "cold"
print("smoke dual OK")
