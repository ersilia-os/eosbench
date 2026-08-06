"""Fetch top scores directly from TDC's own ADMET Benchmark Group leaderboard.

Mirrors fetch_polaris_leaderboard.py's role, but for the "tdc"-provider entries in
tdcommons_leaderboard.json specifically -- TDC's leaderboard (tdcommons.ai) is a genuinely
live, growing thing (new submissions arrive over time via TDC's own Google-Form process),
so a hand-curated snapshot goes stale; this re-fetches the current #1 score for each of the
13 ADMET Benchmark Group *classification* tasks (the 9 regression members are skipped --
prepare_tdcommons.py only prepares binary classification).

Unlike Polaris Hub, TDC's leaderboard needs no scraping trick at all: each dataset's page
(tdcommons.ai/benchmark/admet_group/<slug>/) is a plain server-rendered HTML page with two
ordinary <table>s -- a "Dataset Summary" table (name/task-type/metric/split) and a ranked
"Leaderboard" table -- no JS framework payload to reverse-engineer. Still not an official,
versioned API though: this depends on tdcommons.ai's current page markup and could break if
that changes. Re-run periodically; a moved/renamed heading or table id will show up as
every dataset being skipped, which is a good sign to come back and check the markup.

Usage::

    python scripts/fetch_tdc_leaderboard.py                # all 13 classification tasks
    python scripts/fetch_tdc_leaderboard.py --datasets ames,dili
    python scripts/fetch_tdc_leaderboard.py --no_patch      # write the JSON only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import patch_leaderboard_metadata as patcher

OVERVIEW_URL = "https://tdcommons.ai/benchmark/admet_group/overview/"
DATASET_URL = "https://tdcommons.ai/benchmark/admet_group/{slug}/"
OUT_JSON = Path(__file__).resolve().parent / "tdcommons_leaderboard.json"

_SLUG_RE = re.compile(r"admet_group/([0-9]+[a-z0-9_]*)")
_SUMMARY_RE = re.compile(r"Dataset Summary.*?<table[^>]*>(.*?)</table>", re.S)
_LEADERBOARD_RE = re.compile(r'<table[^>]*id="A"[^>]*>(.*?)</table>', re.S)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td>(.*?)</td>", re.S)
_NAME_RE = re.compile(r">\s*([^<]+?)\s*</a>")
_SCORE_RE = re.compile(r"([\d.]+)(?:\s*<span>&#177;</span>\s*([\d.]+))?")

# TDC's own metric names -> eosbench's catalog convention.
_METRIC_NAMES = {"AUROC": "AUROC", "AUPRC": "AUPRC", "AUPR": "AUPRC"}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", "ignore")


def list_slugs() -> list[str]:
    """All ADMET Benchmark Group page slugs, from the overview page.

    Returns
    -------
    list of str
        Page slugs, e.g. ``"21ames"``.
    """
    html = _get(OVERVIEW_URL)
    seen: list[str] = []
    for slug in _SLUG_RE.findall(html):
        if slug not in seen:
            seen.append(slug)
    return seen


def _first_row_cells(section_html: str) -> list[str]:
    body = re.split(r"</thead>", section_html, maxsplit=1)[-1]
    row = _ROW_RE.search(body)
    if row is None:
        return []
    return _CELL_RE.findall(row.group(1))


def fetch_entry(slug: str) -> dict | None:
    """Fetch one ADMET Benchmark Group page and format its top score as a leaderboard entry.

    Parameters
    ----------
    slug : str
        Page slug, e.g. "21ames".

    Returns
    -------
    dict or None
        A ``tdcommons_leaderboard.json``-shaped entry keyed by the eosbench dataset slug
        (the caller merges the single-key dict in), or ``None`` for a regression task or
        any parse failure (logged), so a markup change fails loudly rather than writing
        garbage.
    """
    html = _get(DATASET_URL.format(slug=slug))

    summary_m = _SUMMARY_RE.search(html)
    if summary_m is None:
        print(f"  [{slug}] skipped: couldn't find the Dataset Summary table")
        return None
    # Row shape: [name (link), unit, size, task, metric, split].
    summary_cells = _first_row_cells(summary_m.group(1))
    if len(summary_cells) < 6:
        print(f"  [{slug}] skipped: Dataset Summary table has an unexpected shape")
        return None
    name_m = _NAME_RE.search(summary_cells[0])
    if name_m is None:
        print(f"  [{slug}] skipped: couldn't parse the dataset name")
        return None
    dataset_slug = name_m.group(1).split(".", 1)[-1].lower()
    task, metric = summary_cells[3].strip(), summary_cells[4].strip()
    split = summary_cells[5].strip().lower()
    if task != "Binary":
        return None  # regression task, out of scope (prepare_tdcommons.py is cls-only)

    lb_m = _LEADERBOARD_RE.search(html)
    if lb_m is None:
        print(f"  [{dataset_slug}] skipped: couldn't find the Leaderboard table")
        return None
    lb_cells = _first_row_cells(lb_m.group(1))
    if len(lb_cells) < 6:
        print(f"  [{dataset_slug}] skipped: Leaderboard table has an unexpected shape")
        return None
    model = lb_cells[1].strip()
    score_m = _SCORE_RE.search(lb_cells[5])
    if score_m is None:
        print(f"  [{dataset_slug}] skipped: couldn't parse the top score")
        return None
    value = float(score_m.group(1))
    std = float(score_m.group(2)) if score_m.group(2) else None

    print(
        f"  [{dataset_slug}] {_METRIC_NAMES.get(metric, metric)} {value:.3f} "
        f"({model!r}{f', ± {std:.3f}' if std is not None else ''})"
    )
    entry = {
        "metric": _METRIC_NAMES.get(metric, metric),
        "value": round(value, 4),
        "split": split,
        "provider": "tdc",
        # Same physical test set as this row's own scaffold holdout (verified byte-for-byte
        # -- see README), but TDC requires 5 independent runs and reports their mean, so
        # it's a different *kind* of number from eosbench's own single-run scaffold_auroc.
        "comparable": "split_only",
        "source": f"{model}, TDC ADMET Benchmark Group leaderboard (5-run avg)",
    }
    if std is not None:
        entry["std"] = round(std, 4)
    return {dataset_slug: entry}


def main() -> None:
    """Entry point: fetch, update ``tdcommons_leaderboard.json``, and sync ``metadata.json``."""
    parser = argparse.ArgumentParser(
        description="Fetch top scores from TDC's own ADMET Benchmark Group leaderboard."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated tdcommons dataset slugs, e.g. ames,dili. Default: all 22 "
        "ADMET Benchmark Group pages (13 classification tasks kept, 9 regression skipped).",
    )
    parser.add_argument(
        "--no_patch",
        action="store_true",
        help="Write tdcommons_leaderboard.json only; skip syncing it into metadata.json.",
    )
    args = parser.parse_args()

    with open(OUT_JSON) as f:
        existing = json.load(f)

    if args.datasets.strip():
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        slugs = list_slugs()
        # Resolve requested dataset slugs back to page slugs by a quick name check.
        page_slugs = []
        for slug in slugs:
            html = _get(DATASET_URL.format(slug=slug))
            m = _SUMMARY_RE.search(html)
            cells = _first_row_cells(m.group(1)) if m else []
            name_m = _NAME_RE.search(cells[0]) if cells else None
            if name_m and name_m.group(1).split(".", 1)[-1].lower() in wanted:
                page_slugs.append(slug)
    else:
        page_slugs = list_slugs()
        print(f"Found {len(page_slugs)} ADMET Benchmark Group page(s).")

    n_updated = 0
    for slug in page_slugs:
        try:
            result = fetch_entry(slug)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  [{slug}] skipped: fetch failed ({e})")
            continue
        if result is None:
            continue
        for dataset_slug, entry in result.items():
            existing[dataset_slug] = {**existing.get(dataset_slug, {}), **entry}
            n_updated += 1

    existing["_snapshot"] = _dt.date.today().isoformat()
    with open(OUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    print(f"\nUpdated {n_updated} entr{'y' if n_updated == 1 else 'ies'} in {OUT_JSON}")

    if not args.no_patch:
        patcher.process("tdcommons", OUT_JSON)
    else:
        print(
            "(--no_patch: metadata.json not synced; run patch_leaderboard_metadata.py "
            "--sources tdcommons to apply)"
        )


if __name__ == "__main__":
    main()
