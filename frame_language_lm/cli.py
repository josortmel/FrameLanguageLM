"""CLI product layer for FrameLanguageLM.

Usage: frame-language-lm <command> [options]

Commands: import, gaps, similar, worth, why, search, setup
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np


def _find_data_dir() -> Path:
    home = Path.home() / ".frame-language-lm"
    if (home / "artifacts" / "model_fp32.onnx").exists():
        return home
    repo = Path(__file__).resolve().parent.parent
    if (repo / "data" / "artifacts" / "model_fp32.onnx").exists():
        return repo / "data"
    return repo / "data"


def _load_sequence(data_dir: Path, profile: str) -> list[int]:
    parquet = data_dir / "user" / f"{profile}_sequence.parquet"
    if not parquet.exists():
        sys.exit(f"Profile not found: {parquet}\nRun: frame-language-lm import --filmaffinity <export.zip>")
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT tconst, indice FROM '{parquet}' ORDER BY timestamp"
    ).fetchall()
    con.close()
    return [int(i) for _, i in rows if i and i > 0]


def _load_full_sequence(data_dir: Path, profile: str):
    from .infer import FrameLM
    parquet = data_dir / "user" / f"{profile}_sequence.parquet"
    if not parquet.exists():
        sys.exit(f"Profile not found: {parquet}")
    full_map = json.loads((data_dir / "full_vocab_map.json").read_text("utf-8"))["tconst_to_idx"]
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT tconst, indice FROM '{parquet}' ORDER BY timestamp"
    ).fetchall()
    con.close()
    seq_warm = [int(i) for _, i in rows if i and i > 0]
    seq_full = [full_map[t] for t, _ in rows if t in full_map]
    return seq_warm, seq_full


def _search_title(catalog: Path, query: str) -> tuple[str, str, int | None]:
    from rapidfuzz import fuzz, process
    con = duckdb.connect(str(catalog), read_only=True)
    rows = con.execute(
        "SELECT tconst, primary_title, original_title, start_year FROM items"
    ).fetchall()
    con.close()
    choices = {}
    for tconst, pt, ot, year in rows:
        label = f"{pt} ({year})" if year else pt
        choices[label] = (tconst, pt, year)
        if ot and ot != pt:
            label2 = f"{ot} ({year})" if year else ot
            choices[label2] = (tconst, ot, year)
    match = process.extractOne(query, choices.keys(), scorer=fuzz.WRatio, score_cutoff=60)
    if not match:
        sys.exit(f"No match for '{query}' in catalog")
    return choices[match[0]]


def _build_filter_query(args) -> str:
    clauses = []
    if getattr(args, "genre", None):
        g = args.genre.replace("'", "''")
        clauses.append(f"genres LIKE '%{g}%'")
    if getattr(args, "decade", None):
        d = args.decade.rstrip("s")
        if d.isdigit():
            start = int(d)
            clauses.append(f"start_year >= {start} AND start_year < {start + 10}")
    if getattr(args, "country", None):
        c = args.country.replace("'", "''")
        clauses.append(f"countries LIKE '%{c}%'")
    if getattr(args, "language", None):
        la = args.language.replace("'", "''")
        clauses.append(f"original_language = '{la}'")
    if getattr(args, "director", None):
        dr = args.director.replace("'", "''")
        clauses.append(f"directors LIKE '%{dr}%'")
    return " AND ".join(clauses) if clauses else ""


def _filter_results(results: list[dict], catalog: Path, filter_q: str) -> list[dict]:
    if not filter_q:
        return results
    tconsts = [r["tconst"] for r in results]
    if not tconsts:
        return results
    con = duckdb.connect(str(catalog), read_only=True)
    placeholders = ",".join(f"'{t}'" for t in tconsts)
    matching = con.execute(
        f"SELECT tconst FROM items WHERE tconst IN ({placeholders}) AND {filter_q}"
    ).fetchall()
    con.close()
    allowed = {r[0] for r in matching}
    return [r for r in results if r["tconst"] in allowed]


def _enrich(results: list[dict], catalog: Path) -> list[dict]:
    if not results:
        return results
    tconsts = [r["tconst"] for r in results]
    placeholders = ",".join(f"'{t}'" for t in tconsts)
    con = duckdb.connect(str(catalog), read_only=True)
    rows = con.execute(
        f"SELECT tconst, primary_title, start_year, title_type, genres, directors, countries "
        f"FROM items WHERE tconst IN ({placeholders})"
    ).fetchall()
    con.close()
    meta = {r[0]: r for r in rows}
    enriched = []
    for r in results:
        m = meta.get(r["tconst"])
        if m:
            r["title"] = m[1]
            r["year"] = m[2]
            r["type"] = m[3]
            r["genres"] = m[4]
            r["directors"] = m[5]
            r["countries"] = m[6]
        enriched.append(r)
    return enriched


def _display_gaps(results: list[dict], title: str) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    table = Table(title=title, show_lines=False, pad_edge=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", min_width=30)
    table.add_column("Year", width=5, justify="center")
    table.add_column("Genre", min_width=15)
    table.add_column("Country", width=7, justify="center")
    table.add_column("Director", min_width=15)
    table.add_column("Score", width=8, justify="right")

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        genres = (r.get("genres") or "").split("|")[0] if r.get("genres") else ""
        directors = (r.get("directors") or "").split("|")[0] if r.get("directors") else ""
        countries = (r.get("countries") or "")[:5]
        score_text = Text(f"{score:+.2f}", style="bold green" if score > 0 else "yellow")
        table.add_row(
            str(i),
            r.get("title", "?"),
            str(r.get("year", "")),
            genres,
            countries,
            directors,
            score_text,
        )
    console.print(table)


def _display_worth(result: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    pop = result.get("population", "?")
    pct = result.get("percentile", 0)
    style = "bold green" if pct >= 90 else "bold yellow" if pct >= 70 else "dim"
    body = (
        f"[bold]{result['title']}[/bold] ({result.get('year', '?')}) [{result.get('type', '?')}]\n\n"
        f"Affinity percentile: [{style}]{pct:.1f}%[/{style}]  (population: {pop})\n"
        f"Raw score: {result.get('score', 0):.4f}"
    )
    console.print(Panel(body, title="Worth", border_style="blue"))


def cmd_setup(args) -> None:
    data_dir = _find_data_dir()
    required = [
        "artifacts/model_fp32.onnx",
        "artifacts/item_embeddings.npy",
        "artifacts/model_full_fp32.onnx",
        "artifacts/item_embeddings_full.npy",
        "artifacts/full_aux.npz",
        "catalog.sqlite",
        "vocab_map.json",
        "full_vocab_map.json",
    ]
    print(f"Data directory: {data_dir}")
    all_ok = True
    for f in required:
        p = data_dir / f
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  OK  {f} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {f}")
            all_ok = False
    if all_ok:
        print("\nAll artifacts present. Ready to use.")
    else:
        print("\nSome artifacts missing. Copy them to the data directory or run setup with HF download (coming soon).")


def cmd_import(args) -> None:
    from .importers import parse_filmaffinity, parse_netflix
    from .importers.matching import Matcher

    data_dir = _find_data_dir()
    events = []
    if args.filmaffinity:
        events += parse_filmaffinity(Path(args.filmaffinity))
    if args.netflix:
        if not args.profile_name:
            sys.exit("--netflix requires --profile-name (Netflix profile name)")
        events += parse_netflix(Path(args.netflix), args.profile_name)
    if not events:
        sys.exit("Nothing to import. Use --filmaffinity <export.zip> or --netflix <ViewingActivity.csv>")

    vocab = json.loads((data_dir / "vocab_map.json").read_text("utf-8"))
    tconst_to_idx = vocab["tconst_to_idx"]

    matcher = Matcher()
    rows, unmatched = [], []
    by_level = {1: 0, 2: 0, 3: 0, 4: 0}

    for ev in events:
        tconst, level = matcher.match(ev["title"], ev["year"], ev["series_hint"])
        if tconst is None:
            unmatched.append(f"{ev['title']} ({ev['year']})")
            continue
        by_level[level] += 1
        idx = tconst_to_idx.get(tconst, -1)
        rows.append((tconst, idx, ev["rating"], ev["timestamp"], ev["source"], ev["title"]))

    rows.sort(key=lambda r: (r[3] is None, r[3]))

    out_dir = data_dir / "user"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"{args.name}_sequence.parquet"
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE seq (tconst VARCHAR, indice BIGINT, rating DOUBLE,
                          timestamp TIMESTAMP, fuente VARCHAR, titulo VARCHAR)
    """)
    con.executemany("INSERT INTO seq VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY seq TO '{parquet}' (FORMAT PARQUET)")
    con.close()

    n_in_vocab = sum(1 for r in rows if r[1] > 0)
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold green]Imported {len(rows)} titles[/bold green] "
                  f"({n_in_vocab} in warm vocab, {len(rows) - n_in_vocab} cold)")
    console.print(f"Unmatched: {len(unmatched)}")
    console.print(f"Saved to: {parquet}")
    if unmatched:
        console.print("\n[dim]Unmatched titles:[/dim]")
        for t in unmatched[:20]:
            console.print(f"  - {t}")
        if len(unmatched) > 20:
            console.print(f"  ... and {len(unmatched) - 20} more")


def cmd_gaps(args) -> None:
    data_dir = _find_data_dir()
    catalog = data_dir / "catalog.sqlite"
    from .infer import FrameLM

    lm = FrameLM(data_dir / "artifacts", catalog, data_dir)
    seq_warm, seq_full = _load_full_sequence(data_dir, args.name)

    filter_q = _build_filter_query(args)
    k = args.top

    title_type = getattr(args, "type", None)

    if title_type == "series":
        result = lm.gaps_series(seq_full, k * 3)
        warm = _enrich(result["warm"], catalog)
        cold = _enrich(result["cold"], catalog)
        if filter_q:
            warm = _filter_results(warm, catalog, filter_q)
            cold = _filter_results(cold, catalog, filter_q)
        _display_gaps(warm[:k], f"Series — warm block ({len(seq_warm)} events)")
        _display_gaps(cold[:k], f"Series — cold block ({len(seq_full)} events)")
    elif title_type == "doc":
        result = lm.gaps_docs(seq_full, k * 3)
        warm = _enrich(result["warm"], catalog)
        cold = _enrich(result["cold"], catalog)
        if filter_q:
            warm = _filter_results(warm, catalog, filter_q)
            cold = _filter_results(cold, catalog, filter_q)
        _display_gaps(warm[:k], "Documentaries — warm block")
        _display_gaps(cold[:k], "Documentaries — cold block")
    else:
        results = lm.gaps_movies(seq_warm, k * 3)
        results = _enrich(results, catalog)
        if filter_q:
            results = _filter_results(results, catalog, filter_q)
        _display_gaps(results[:k], f"Movies — top {min(k, len(results))} gaps ({len(seq_warm)} warm events)")


def cmd_similar(args) -> None:
    data_dir = _find_data_dir()
    catalog = data_dir / "catalog.sqlite"
    from .infer import FrameLM

    tconst, title, year = _search_title(catalog, args.title)
    lm = FrameLM(data_dir / "artifacts", catalog, data_dir)
    neighbors = lm.neighbors(tconst, args.top)
    neighbors = _enrich(neighbors, catalog)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]Neighbors of[/bold] {title} ({year or '?'})\n")
    table = Table(show_lines=False, pad_edge=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", min_width=30)
    table.add_column("Year", width=5, justify="center")
    table.add_column("Genre", min_width=15)
    table.add_column("Similarity", width=10, justify="right")
    for i, n in enumerate(neighbors, 1):
        genres = (n.get("genres") or "").split("|")[0]
        table.add_row(str(i), n["title"], str(n.get("year", "")), genres, f"{n['sim']:.3f}")
    console.print(table)


def cmd_worth(args) -> None:
    data_dir = _find_data_dir()
    catalog = data_dir / "catalog.sqlite"
    from .infer import FrameLM

    tconst, title, year = _search_title(catalog, args.title)
    lm = FrameLM(data_dir / "artifacts", catalog, data_dir)
    seq_warm, seq_full = _load_full_sequence(data_dir, args.name)

    full_map = lm.tconst_to_idx
    idx = full_map.get(tconst)
    if idx is not None and lm.aux["is_cold"][idx]:
        seq = seq_full
    else:
        seq = seq_warm

    result = lm.worth(tconst, seq)
    _display_worth(result)


def cmd_why(args) -> None:
    data_dir = _find_data_dir()
    catalog = data_dir / "catalog.sqlite"
    from .infer import FrameLM

    tconst, title, year = _search_title(catalog, args.title)
    lm = FrameLM(data_dir / "artifacts", catalog, data_dir)
    seq_warm, seq_full = _load_full_sequence(data_dir, args.name)

    full_map = lm.tconst_to_idx
    idx = full_map.get(tconst)
    is_cold = idx is not None and lm.aux["is_cold"][idx]
    seq = seq_full if is_cold else seq_warm

    result = lm.worth(tconst, seq)
    neighbors = lm.neighbors(tconst, 10)
    neighbors = _enrich(neighbors, catalog)

    user_tconsts = set()
    parquet = data_dir / "user" / f"{args.name}_sequence.parquet"
    if parquet.exists():
        con = duckdb.connect()
        rows = con.execute(f"SELECT tconst FROM '{parquet}'").fetchall()
        con.close()
        user_tconsts = {r[0] for r in rows}

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    pct = result.get("percentile", 0)
    pop = result.get("population", "?")
    style = "bold green" if pct >= 90 else "bold yellow" if pct >= 70 else "dim"

    console.print(f"\n[bold]Why {title} ({year or '?'})?[/bold]\n")
    console.print(f"Affinity percentile: [{style}]{pct:.1f}%[/{style}]  (population: {pop})")
    console.print(f"Raw score: {result.get('score', 0):.4f}\n")

    console.print("[bold]Nearest neighbors in embedding space:[/bold]")
    table = Table(show_lines=False, pad_edge=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", min_width=30)
    table.add_column("Year", width=5, justify="center")
    table.add_column("Sim", width=8, justify="right")
    table.add_column("In profile", width=10, justify="center")
    for i, n in enumerate(neighbors, 1):
        in_profile = "[green]yes[/green]" if n["tconst"] in user_tconsts else "[dim]no[/dim]"
        table.add_row(str(i), n["title"], str(n.get("year", "")), f"{n['sim']:.3f}", in_profile)
    console.print(table)
    in_count = sum(1 for n in neighbors if n["tconst"] in user_tconsts)
    console.print(f"\n[dim]{in_count}/{len(neighbors)} neighbors are in your profile — "
                  f"the model sees affinity through these connections.[/dim]")


def cmd_search(args) -> None:
    data_dir = _find_data_dir()
    catalog = data_dir / "catalog.sqlite"

    filter_q = _build_filter_query(args)
    if not filter_q:
        sys.exit("At least one filter required: --genre, --decade, --country, --language, --director")

    con = duckdb.connect(str(catalog), read_only=True)
    rows = con.execute(
        f"SELECT tconst, primary_title, start_year, title_type, genres, directors, countries, "
        f"imdb_rating, num_votes FROM items WHERE {filter_q} "
        f"ORDER BY num_votes DESC LIMIT {args.top}"
    ).fetchall()
    con.close()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Search results ({len(rows)} titles)", show_lines=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", min_width=30)
    table.add_column("Year", width=5, justify="center")
    table.add_column("Type", width=10)
    table.add_column("Genre", min_width=12)
    table.add_column("Director", min_width=15)
    table.add_column("Rating", width=6, justify="right")
    table.add_column("Votes", width=10, justify="right")
    for i, r in enumerate(rows, 1):
        genres = (r[4] or "").split("|")[0]
        directors = (r[5] or "").split("|")[0]
        rating = f"{r[7]:.1f}" if r[7] else ""
        votes = f"{r[8]:,}" if r[8] else ""
        table.add_row(str(i), r[1], str(r[2] or ""), r[3], genres, directors, rating, votes)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="frame-language-lm",
        description="A language model whose vocabulary is films",
    )
    sub = parser.add_subparsers(dest="command")

    # setup
    p_setup = sub.add_parser("setup", help="Verify/download artifacts")
    p_setup.set_defaults(func=cmd_setup)

    # import
    p_import = sub.add_parser("import", help="Import user viewing history")
    p_import.add_argument("--filmaffinity", help="FilmAffinity GDPR export ZIP")
    p_import.add_argument("--netflix", help="Netflix ViewingActivity.csv")
    p_import.add_argument("--profile-name", help="Netflix profile name (required with --netflix)")
    p_import.add_argument("--name", default="pepe", help="Profile name for saved sequence")
    p_import.set_defaults(func=cmd_import)

    # gaps
    p_gaps = sub.add_parser("gaps", help="Find your viewing gaps")
    p_gaps.add_argument("--top", type=int, default=50, help="Number of results")
    p_gaps.add_argument("--type", choices=["movie", "series", "doc"], default="movie")
    p_gaps.add_argument("--genre", help="Filter by genre")
    p_gaps.add_argument("--decade", help="Filter by decade (e.g. 1990s)")
    p_gaps.add_argument("--country", help="Filter by country code")
    p_gaps.add_argument("--language", help="Filter by original language")
    p_gaps.add_argument("--director", help="Filter by director name")
    p_gaps.add_argument("--name", default="pepe", help="Profile name")
    p_gaps.set_defaults(func=cmd_gaps)

    # similar
    p_sim = sub.add_parser("similar", help="Find titles similar to a given one")
    p_sim.add_argument("title", help="Title to search for")
    p_sim.add_argument("--top", type=int, default=10)
    p_sim.set_defaults(func=cmd_similar)

    # worth
    p_worth = sub.add_parser("worth", help="Is this title worth watching for you?")
    p_worth.add_argument("title", help="Title to evaluate")
    p_worth.add_argument("--name", default="pepe", help="Profile name")
    p_worth.set_defaults(func=cmd_worth)

    # why
    p_why = sub.add_parser("why", help="Explain why a title is recommended")
    p_why.add_argument("title", help="Title to explain")
    p_why.add_argument("--name", default="pepe", help="Profile name")
    p_why.set_defaults(func=cmd_why)

    # search
    p_search = sub.add_parser("search", help="Search the catalog")
    p_search.add_argument("--genre", help="Filter by genre")
    p_search.add_argument("--decade", help="Filter by decade (e.g. 1990s)")
    p_search.add_argument("--country", help="Filter by country code")
    p_search.add_argument("--language", help="Filter by original language")
    p_search.add_argument("--director", help="Filter by director name")
    p_search.add_argument("--type", choices=["movie", "tvSeries", "tvMiniSeries"])
    p_search.add_argument("--top", type=int, default=20)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)
