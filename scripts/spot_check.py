"""Fase 1: spot-check manual del catalogo ensamblado."""

import duckdb

con = duckdb.connect("data/catalog.sqlite")
cols = [r[1] for r in con.execute("PRAGMA table_info(items)").fetchall()]
print("cols:", cols)

KNOWN = [
    "tt0133093",  # The Matrix
    "tt0245429",  # Spirited Away
    "tt0903747",  # Breaking Bad
    "tt0060827",  # Persona
    "tt7286456",  # Joker
    "tt0084787",  # The Thing
    "tt1856101",  # Blade Runner 2049
    "tt0050986",  # Wild Strawberries
    "tt2543164",  # Arrival
    "tt0105288",  # Reservoir Dogs? (check)
]

q = """SELECT tconst, primary_title, start_year, directors, original_language,
       countries, budget FROM items WHERE tconst = ?"""
for t in KNOWN:
    print(con.execute(q, [t]).fetchone())

print("\naleatorios:")
rows = con.execute(
    """SELECT tconst, primary_title, start_year, directors, original_language,
       countries FROM items USING SAMPLE 10"""
).fetchall()
for r in rows:
    print(r)
