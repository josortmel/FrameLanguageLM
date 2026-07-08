"""Importador del export oficial RGPD de FilmAffinity (ZIP de HTML).

Extrae de html/movie-ratings.html: (titulo_es, año, nota /5, timestamp).
La nota original es /10; se divide entre 2 para alinear con MovieLens.
"""

import html
import re
import zipfile
from datetime import datetime
from pathlib import Path

ROW_RE = re.compile(
    r'<div class="user-rating">(\d+)</div></td>\s*'
    r"<td>(.*?)</td>\s*"
    r"<td><em>(.*?)</em>",
    re.DOTALL,
)
TITLE_YEAR_RE = re.compile(r"^(.*)\s+\((\d{4})\)\s*$", re.DOTALL)

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
DATE_RE = re.compile(r"(\d{1,2}) de (\w+) de (\d{4}),?\s*(\d{1,2}):(\d{2})")


def _parse_date(text: str) -> datetime | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    d, mes, y, hh, mm = m.groups()
    month = MESES.get(mes.lower())
    if not month:
        return None
    return datetime(int(y), month, int(d), int(hh), int(mm))


def parse_filmaffinity(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("movie-ratings.html"))
        raw = z.read(name).decode("utf-8", errors="replace")

    events = []
    for rating10, td_title, td_date in ROW_RE.findall(raw):
        text = html.unescape(td_title).strip()
        m = TITLE_YEAR_RE.match(text)
        title, year = (m.group(1).strip(), int(m.group(2))) if m else (text, None)
        ts = _parse_date(html.unescape(td_date))
        events.append(
            {
                "title": title,
                "year": year,
                "rating": int(rating10) / 2.0,
                "timestamp": ts,
                "source": "filmaffinity",
                "series_hint": None,
            }
        )
    return events
