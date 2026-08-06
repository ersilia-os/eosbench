"""Backfill leaderboard fields into existing metadata.json from the curated JSONs.

``prepare_family`` writes the leaderboard fields for fresh runs; this patches them into
already-generated metadata (family level + each column block) without re-featurising, so a
leaderboard edit (new score, provider, split) goes live without re-running the prep
pipeline. Idempotent.

Provenance: each leaderboard JSON entry carries a ``provider`` (e.g. ``polaris`` for the
Polaris-hosted TDC ADMET leaderboard, ``tdc`` for a TDC-native fallback, ``moleculenet`` for
the MoleculeNet paper). Precedence for tdcommons is Polaris first, TDC as fallback — encoded
directly in scripts/tdcommons_leaderboard.json.

The ``polaris`` source's own leaderboard JSON (``polaris_leaderboard.json``) is maintained by
``fetch_polaris_leaderboard.py``, not hand-curated: it live-fetches the current top score
straight from the Hub, since the Polaris Hub leaderboard for a benchmark keeps gaining new
submissions over time and is *not* the same thing as the (comparatively static)
``tdcommons_leaderboard.json`` "polaris"-provider entries, which mirror what TDC's own ADMET
Benchmark Group snapshot showed at curation time. Re-run that script to refresh it.

Usage::

    python scripts/patch_leaderboard_metadata.py                 # all known sources
    python scripts/patch_leaderboard_metadata.py --sources tdcommons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _prepare_common as common

# source -> curated leaderboard JSON (keyed by dataset/family slug)
SOURCES = {
    "moleculenet": Path(__file__).resolve().parent / "moleculenet_leaderboard.json",
    "tdcommons": Path(__file__).resolve().parent / "tdcommons_leaderboard.json",
    "polaris": Path(__file__).resolve().parent / "polaris_leaderboard.json",
}
_FIELDS = ("metric", "value", "std", "split", "provider", "source", "fetched_at", "comparable")


def _load(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _entry_fields(entry: dict | None) -> dict:
    entry = entry or {}
    return {f"leaderboard_{k}": entry.get(k) for k in _FIELDS}


def _patch(meta_path: Path, fields: dict) -> bool:
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update(fields)
    for block in meta.get("columns", {}).values():
        block.update(fields)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return True


def process(source: str, json_path: Path) -> None:
    """Patch every task (classification, regression, ...) this source has bundled data
    for -- moleculenet's leaderboard JSON covers both, so this must not assume just one."""
    leaderboard = _load(json_path)
    source_root = common.PKG_DATA_ROOT / source
    if not source_root.exists():
        print(f"  [{source}] no bundled data; skipping")
        return
    n_families = n_lb = 0
    for task_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        task = task_dir.name
        families = sorted(p.name for p in task_dir.iterdir() if (p / "metadata.json").exists())
        for family in families:
            fields = _entry_fields(leaderboard.get(family))
            for root in (common.PKG_DATA_ROOT, common.DATA_ROOT):
                _patch(root / source / task / family / "metadata.json", fields)
            if fields["leaderboard_value"] is not None:
                n_lb += 1
        n_families += len(families)
    print(
        f"  [{source}] patched {n_families} families ({n_lb} with a leaderboard score)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill leaderboard fields into metadata."
    )
    parser.add_argument(
        "--sources", type=str, default=None, help="Comma-separated (default: all)."
    )
    args = parser.parse_args()
    want = (
        {s.strip() for s in args.sources.split(",")} if args.sources else set(SOURCES)
    )
    for source, json_path in SOURCES.items():
        if source in want:
            process(source, json_path)
    print("\nDone. Re-sync data/ to S3 if you maintain the upload copy.")


if __name__ == "__main__":
    main()
