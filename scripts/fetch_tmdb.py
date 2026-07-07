"""Fase 1.3: enriquecimiento TMDB, reanudable.

Uso: uv run python scripts/fetch_tmdb.py [--limit N]
Entrada: data/interim/vocab.parquet + TMDB_READ_TOKEN en .env
Salida: data/interim/tmdb.jsonl (append-only; relanzar salta lo ya hecho)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import duckdb
import httpx

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "interim" / "vocab.parquet"
OUT = ROOT / "data" / "interim" / "tmdb.jsonl"
BASE = "https://api.themoviedb.org/3"
CONCURRENCY = 20


def read_token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TMDB_READ_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("TMDB_READ_TOKEN no encontrado en .env")


async def get_json(client: httpx.AsyncClient, url: str) -> dict | None:
    """None = 404. Reintenta 429/5xx/errores de red con backoff."""
    for attempt in range(6):
        try:
            r = await client.get(url)
        except httpx.HTTPError:
            await asyncio.sleep(2**attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = float(r.headers.get("Retry-After", 2**attempt))
            await asyncio.sleep(retry_after)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"agotados reintentos: {url}")


def extract(detail: dict, media_type: str) -> dict:
    kw_key = "keywords" if media_type == "movie" else "results"
    return {
        "tmdb_id": detail["id"],
        "media_type": media_type,
        "original_language": detail.get("original_language"),
        "production_countries": [c["iso_3166_1"] for c in detail.get("production_countries", [])],
        "budget": detail.get("budget") if media_type == "movie" else None,
        "keywords": [k["name"] for k in detail.get("keywords", {}).get(kw_key, [])],
        "poster_path": detail.get("poster_path"),
        "popularity": detail.get("popularity"),
    }


async def fetch_one(client: httpx.AsyncClient, row: dict) -> dict:
    tconst, title_type, tmdb_id = row["tconst"], row["titleType"], row["tmdb_id_links"]
    if tmdb_id is not None and title_type == "movie":
        media, mid = "movie", int(tmdb_id)
    else:
        found = await get_json(client, f"/find/{tconst}?external_source=imdb_id")
        if found is None:
            return {"tconst": tconst, "error": "not_found"}
        if title_type != "movie" and found.get("tv_results"):
            media, mid = "tv", found["tv_results"][0]["id"]
        elif found.get("movie_results"):
            media, mid = "movie", found["movie_results"][0]["id"]
        elif found.get("tv_results"):
            media, mid = "tv", found["tv_results"][0]["id"]
        else:
            return {"tconst": tconst, "error": "not_found"}
    detail = await get_json(client, f"/{media}/{mid}?append_to_response=keywords")
    if detail is None:
        return {"tconst": tconst, "error": "not_found"}
    return {"tconst": tconst} | extract(detail, media)


async def main(limit: int | None) -> None:
    done: set[str] = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {json.loads(line)["tconst"] for line in f if line.strip()}

    rows = duckdb.sql(
        f"SELECT tconst, titleType, tmdb_id_links FROM read_parquet('{VOCAB.as_posix()}')"
    ).fetchall()
    pending = [
        {"tconst": t, "titleType": tt, "tmdb_id_links": mid}
        for t, tt, mid in rows
        if t not in done
    ]
    if limit is not None:
        pending = pending[:limit]
    print(f"ya hechos: {len(done):,} | pendientes en esta pasada: {len(pending):,}")
    if not pending:
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counter = {"n": 0, "err": 0}

    async with httpx.AsyncClient(
        base_url=BASE,
        headers={"Authorization": f"Bearer {read_token()}"},
        timeout=30,
    ) as client:
        with OUT.open("a", encoding="utf-8") as out:

            async def worker(row: dict) -> None:
                async with sem:
                    rec = await fetch_one(client, row)
                async with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counter["n"] += 1
                    counter["err"] += "error" in rec
                    if counter["n"] % 500 == 0:
                        out.flush()
                        print(f"  {counter['n']:,}/{len(pending):,} ({counter['err']} errores)")

            await asyncio.gather(*(worker(r) for r in pending))

    print(f"completado: {counter['n']:,} registros, {counter['err']} sin match")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main(ap.parse_args().limit))
