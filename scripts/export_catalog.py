"""Export the eosbench catalog to a JSON file for the GitHub Pages browser.

Reads the bundled dataset metadata (the same source `eosbench catalog` uses) across every
source and both tasks, HEAD-probes each dataset's ``data.csv`` on the public S3 bucket to mark
it available/pending, and writes a single ``catalog.json`` consumed by ``site/index.html``.

Discovery is driven entirely by the committed metadata; only the availability flag is read from
S3. The CI workflow runs this on every push and on a daily cron so the dataset list tracks the
repo and the availability badges track live S3.

Usage::

    python scripts/export_catalog.py                       # -> site/catalog.json
    python scripts/export_catalog.py --out path/to.json
    python scripts/export_catalog.py --no-check            # skip S3 HEAD probes (all pending)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from eosbench import get_catalog, DatasetInfo
from eosbench.dataset import S3_BASE, catalog_columns

TASKS = ("classification", "regression")

# Union of every column that can appear across tasks/views, so the JS renderer is uniform.
_UNION_KEYS = []
for _task in TASKS:
    for _col in catalog_columns(_task):
        if _col not in _UNION_KEYS:
            _UNION_KEYS.append(_col)


def _head_ok(url: str, timeout: float = 15.0) -> bool:
    """True if a HEAD request to ``url`` returns HTTP 200 (object exists on S3)."""
    try:
        with urlopen(Request(url, method="HEAD"), timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError, ValueError):
        return False


def _data_url(source: str, task: str, dataset: str) -> str:
    return f"{S3_BASE}/{source}/{task}/{dataset}/data.csv"


def _clean(v):
    """Coerce numpy scalars to Python and missing/non-finite floats to None (valid JSON)."""
    if hasattr(v, "item"):  # numpy int64/float64/bool_ -> native Python
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def build_catalog(head_check=_head_ok) -> dict:
    """Build the catalog payload. ``head_check(url) -> bool`` decides availability (injectable)."""
    records: list[dict] = []
    for task in TASKS:
        df = get_catalog(task=task)  # all sources; empty frame if none for this task
        for row in df.to_dict(orient="records"):
            source, name = row["source"], row["name"]
            rec = {key: _clean(row.get(key)) for key in _UNION_KEYS}
            rec["columns"] = DatasetInfo(source, task, name).columns
            rec["data_url"] = _data_url(source, task, name)
            rec["fetch_cmd"] = (
                f"eosbench fetch --source {source} --task {task} --dataset {name}"
            )
            rec["available"] = bool(head_check(rec["data_url"]))
            records.append(rec)

    records.sort(key=lambda r: (r["source"], r["task"], r["name"]))
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "s3_base": S3_BASE,
        "n_datasets": len(records),
        "n_available": sum(1 for r in records if r["available"]),
        "columns": _UNION_KEYS,
        "datasets": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the eosbench catalog to JSON.")
    parser.add_argument(
        "--out",
        type=str,
        default="site/catalog.json",
        help="Output path (default: site/catalog.json).",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip S3 availability HEAD probes (mark all pending).",
    )
    args = parser.parse_args()

    payload = build_catalog(
        head_check=(lambda _url: False) if args.no_check else _head_ok
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(
        f"Wrote {payload['n_datasets']} datasets "
        f"({payload['n_available']} available on S3) -> {out}"
    )


if __name__ == "__main__":
    main()
