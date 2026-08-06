"""Fetch live leaderboard scores directly from the Polaris Hub, for its TDC-mirror benchmarks.

The Polaris Hub's own leaderboard for a benchmark keeps gaining new submissions over time --
it is a genuinely different, moving thing from the (comparatively static) curated references
in tdcommons_leaderboard.json's "polaris"-provider entries, which were a snapshot of what TDC's
own ADMET Benchmark Group page showed at curation time. This script reads the current top score
straight from the Hub instead, for the ``polaris`` source's own families (see
prepare_polaris.py).

There is no anonymous, documented way to do this. polaris-lib's client and the Hub's own
documented REST API require an authenticated login just to read one submitted result's score
(``GET /v1/result/{id}`` returns 401 without a bearer token) -- there is no public "list
leaderboard scores" endpoint. The public benchmark page itself
(``polarishub.io/benchmarks/{owner}/{slug}``) embeds the same data in its server-rendered React
payload though, sorted best-first, and *that* is readable anonymously -- this scrapes that
payload instead of using polaris-lib at all (no ``prepare-polaris`` extra needed; stdlib only).

This is NOT an officially documented API: it depends on Polaris's current Next.js internal
serialization format and can break silently if that changes. Treat it as a best-effort refresh
you re-run occasionally, not a guaranteed-stable integration -- spot-check a row or two against
the website after a big jump before trusting it.

Usage::

    python scripts/fetch_polaris_leaderboard.py                          # all tdcommons/* classification benchmarks
    python scripts/fetch_polaris_leaderboard.py --datasets tdcommons/ames,tdcommons/dili
    python scripts/fetch_polaris_leaderboard.py --no_patch                # write the JSON only, skip syncing metadata.json
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

HUB_API = "https://polarishub.io/api"
HUB_WEB = "https://polarishub.io"
OUT_JSON = Path(__file__).resolve().parent / "polaris_leaderboard.json"

# One entry per submitted result on a benchmark's page, embedded in its server-rendered
# payload as an HTML-escaped JSON fragment (quotes escaped as \"). Order on the page is
# already best-first, but we re-sort defensively rather than rely on that.
_RESULT_RE = re.compile(
    r'\\"name\\":\\"(?P<name>.*?)\\".*?'
    r'\\"scores\\":\{(?P<scores>.*?)\}.*?'
    r'\\"testSet\\":\\"(?P<testset>.*?)\\".*?'
    r'\\"target\\":\\"(?P<target>.*?)\\".*?'
    r'\\"actions\\":\{\\"url\\":\\"(?P<resulturl>.*?)\\"\}.*?'
    r'\\"isCertified\\":(?P<cert>true|false)'
)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def list_benchmark_ids(owner_prefix: str) -> list[str]:
    """All Hub benchmark ids (v2 + v1, paginated) whose owner matches ``owner_prefix``."""
    ids: list[str] = []
    for version in ("v2", "v1"):
        offset = 0
        while True:
            batch = _get_json(f"{HUB_API}/{version}/benchmark?limit=100&offset={offset}")[
                "data"
            ]
            if not batch:
                break
            ids.extend(b["artifactId"] for b in batch)
            if len(batch) < 100:
                break
            offset += 100
    return sorted({b for b in ids if b.startswith(f"{owner_prefix}/")})


def slugify(benchmark_id: str) -> str:
    return benchmark_id.split("/")[-1].lower()


def fetch_top_result(owner: str, slug: str) -> dict | None:
    """Fetch this benchmark's declared metric and its current best submitted score.

    Returns ``None`` if the benchmark isn't classification (no ``nClasses``) or has no
    submitted results yet.
    """
    detail = _get_json(f"{HUB_API}/v1/benchmark/{owner}/{slug}")
    if not detail.get("nClasses"):
        return None  # regression benchmark
    metric = detail.get("mainMetric")

    req = urllib.request.Request(
        f"{HUB_WEB}/benchmarks/{owner}/{slug}", headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "ignore")

    results = []
    for m in _RESULT_RE.finditer(html):
        scores_raw = "{" + m.group("scores").replace('\\"', '"') + "}"
        try:
            scores = json.loads(scores_raw)
        except json.JSONDecodeError:
            continue
        if metric in scores:
            results.append({"name": m.group("name"), "value": scores[metric]})
    if not results:
        return None
    best = max(results, key=lambda r: r["value"])
    return {"metric": metric, "value": best["value"], "model_name": best["name"]}


def build_entry(owner: str, slug: str, today: str) -> dict | None:
    try:
        top = fetch_top_result(owner, slug)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [{owner}/{slug}] skipped: fetch failed ({e})")
        return None
    if top is None:
        print(f"  [{owner}/{slug}] skipped: no classification results found")
        return None
    metric_name = {"roc_auc": "AUROC", "pr_auc": "AUPRC"}.get(
        top["metric"], top["metric"].upper()
    )
    print(f"  [{owner}/{slug}] {metric_name} {top['value']:.3f} ({top['model_name']!r})")
    return {
        "metric": metric_name,
        "value": round(float(top["value"]), 4),
        "split": "scaffold",
        "provider": "polaris",
        # Polaris Hub's own result schema (ResultRecords: test_set/target_label/scores) has
        # no seed-count or std field at all -- a submission is always a single reported
        # number, same as eosbench's own single-run scaffold_auroc. So this is comparable
        # to a local single-run result on the identical split, unlike a tdc-provider entry
        # (a 5-run average) -- see scripts/tdcommons_leaderboard.json's _comment.
        "comparable": "yes",
        "source": f"{top['model_name']}, live Polaris Hub leaderboard",
        "fetched_at": today,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live leaderboard scores from the Polaris Hub."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated benchmark ids (owner/slug). Default: auto-discover tdcommons/*.",
    )
    parser.add_argument(
        "--no_patch",
        action="store_true",
        help="Write polaris_leaderboard.json only; skip syncing it into metadata.json.",
    )
    args = parser.parse_args()

    if args.datasets.strip():
        ids = [b.strip() for b in args.datasets.split(",") if b.strip()]
    else:
        print("Discovering tdcommons/* benchmarks on the Polaris Hub...")
        ids = list_benchmark_ids("tdcommons")
        print(f"  found {len(ids)} benchmark(s)")

    existing = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    today = _dt.date.today().isoformat()
    for bid in ids:
        owner, _, slug = bid.partition("/")
        entry = build_entry(owner, slug, today)
        if entry is not None:
            existing[slugify(bid)] = entry

    out = {
        "_comment": (
            "Live-fetched Polaris Hub leaderboard top scores for the `polaris` source's "
            "tdcommons-mirror benchmarks. Maintained by fetch_polaris_leaderboard.py -- do "
            "not hand-edit; re-run that script to refresh. Each entry's 'fetched_at' records "
            "when THAT score was captured; the Hub leaderboard gains new submissions over "
            "time, so re-run periodically rather than treating this as a one-time snapshot."
        ),
        **dict(sorted(existing.items())),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(existing)} entries to {OUT_JSON}")

    if not args.no_patch:
        patcher.process("polaris", OUT_JSON)
    else:
        print("(--no_patch: metadata.json not synced; run patch_leaderboard_metadata.py "
              "--sources polaris to apply)")


if __name__ == "__main__":
    main()
