"""Fase 0: descarga reproducible del corpus crudo a data/raw/.

Uso: uv run python scripts/download_data.py
Solo stdlib. Reanudable: si un fichero ya existe con el tamano remoto, se salta.
"""

import gzip
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

MOVIELENS = ["https://files.grouplens.org/datasets/movielens/ml-32m.zip"]

IMDB = [
    f"https://datasets.imdbws.com/{name}.tsv.gz"
    for name in [
        "title.basics",
        "title.crew",
        "title.principals",
        "title.ratings",
        "title.akas",
        "title.episode",
        "name.basics",
    ]
]


def tmdb_export_urls() -> list[str]:
    # Los daily exports se publican hacia las 8:00 UTC; si el de hoy aun no
    # existe, el descargador cae al de ayer.
    urls = []
    for delta in (0, 1):
        d = datetime.now(timezone.utc) - timedelta(days=delta)
        stamp = d.strftime("%m_%d_%Y")
        urls.append(
            [
                f"https://files.tmdb.org/p/exports/movie_ids_{stamp}.json.gz",
                f"https://files.tmdb.org/p/exports/tv_series_ids_{stamp}.json.gz",
            ]
        )
    return urls


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            size = r.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        return None


def download(url: str, dest: Path) -> bool:
    size = remote_size(url)
    if size is None:
        print(f"  SKIP (no accesible): {url}")
        return False
    if dest.exists() and dest.stat().st_size == size:
        print(f"  ok (ya existe): {dest.name} ({size / 1e6:.0f} MB)")
        return True
    print(f"  descargando {dest.name} ({size / 1e6:.0f} MB)...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return True


def verify() -> int:
    print("\n== Verificacion fase 0 ==")
    failures = 0

    ml_zip = RAW / "ml-32m.zip"
    with zipfile.ZipFile(ml_zip) as z:
        names = z.namelist()
        assert "ml-32m/links.csv" in names, "links.csv ausente"
        with z.open("ml-32m/ratings.csv") as f:
            n = sum(1 for _ in f) - 1  # header
    expected = 32_000_204
    status = "OK" if n == expected else "FALLO"
    if n != expected:
        failures += 1
    print(f"  [{status}] ratings.csv: {n:,} filas (esperado {expected:,})")
    print("  [OK] links.csv presente")

    basics = RAW / "title.basics.tsv.gz"
    with gzip.open(basics, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        row = f.readline().rstrip("\n").split("\t")
    needed = {"tconst", "titleType", "primaryTitle", "startYear", "genres"}
    missing = needed - set(header)
    if missing or len(row) != len(header):
        failures += 1
        print(f"  [FALLO] title.basics: columnas faltantes {missing}")
    else:
        print(f"  [OK] title.basics parsea ({len(header)} columnas)")

    return failures


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    print("== MovieLens ==")
    for url in MOVIELENS:
        if not download(url, RAW / url.rsplit("/", 1)[1]):
            sys.exit(f"critico: no se pudo descargar {url}")

    print("== IMDb ==")
    for url in IMDB:
        if not download(url, RAW / url.rsplit("/", 1)[1]):
            sys.exit(f"critico: no se pudo descargar {url}")

    print("== TMDB daily exports ==")
    for day_urls in tmdb_export_urls():
        results = [download(u, RAW / u.rsplit("/", 1)[1]) for u in day_urls]
        if all(results):
            break
    else:
        sys.exit("critico: ningun daily export de TMDB accesible (hoy ni ayer)")

    if verify():
        sys.exit("verificacion con fallos")
    print("\nFase 0 completa.")


if __name__ == "__main__":
    main()
