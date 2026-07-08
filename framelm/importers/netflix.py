"""Importador de ViewingActivity.csv del export RGPD de Netflix.

Filtra por perfil, descarta previews (<15 min y Supplemental Video Type),
colapsa episodios a nivel de serie y conserva el PRIMER visionado por
titulo como evento de secuencia. Sin nota (rating 0 = sin-rating).
"""

import csv
import re
from datetime import datetime
from pathlib import Path

MIN_SECONDS = 15 * 60

# marcadores de episodio en titulos localizados de Netflix ES
SEASON_MARKERS = re.compile(
    r": (Temporada|Parte|Cap[ií]tulo|Volumen|Season|Miniserie|Serie limitada) "
)
EPISODE_SUFFIX = re.compile(r"\s*\(Episodio \d+\)\s*$")


def _duration_seconds(text: str) -> int:
    try:
        h, m, s = text.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except ValueError:
        return 0


def _collapse_title(title: str) -> tuple[str, bool]:
    """Devuelve (titulo colapsado, es_serie)."""
    m = SEASON_MARKERS.search(title)
    if m:
        return title[: m.start()].strip(), True
    if EPISODE_SUFFIX.search(title):
        # episodio sin marcador de temporada: corta en el primer ': '
        base = title.split(": ", 1)[0].strip()
        return base, True
    return title.strip(), False


def parse_netflix(csv_path: Path, profile: str) -> list[dict]:
    first_watch: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["Profile Name"] != profile:
                continue
            if row.get("Supplemental Video Type", "").strip():
                continue  # trailers, teasers, recaps
            if _duration_seconds(row["Duration"]) < MIN_SECONDS:
                continue
            title, is_series = _collapse_title(row["Title"])
            if not title:
                continue
            ts = datetime.strptime(row["Start Time"], "%Y-%m-%d %H:%M:%S")
            prev = first_watch.get(title)
            if prev is None or ts < prev["timestamp"]:
                first_watch[title] = {
                    "title": title,
                    "year": None,
                    "rating": 0.0,
                    "timestamp": ts,
                    "source": "netflix",
                    "series_hint": is_series,
                }
    return list(first_watch.values())
