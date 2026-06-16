"""Inject a per-column scientific ``description`` into MoleculeNet metadata.json files.

Reads the one-liners from ``scripts/moleculenet_descriptions.py`` and writes a
``description`` (+ ``description_source``) into every column block of each MoleculeNet
family, in both the bundled package metadata and the local ``data/`` copy. Idempotent —
safe to re-run. Does not recompute features, splits, or baselines.

Usage::

    pip install -e ".[prepare]"
    python scripts/add_moleculenet_descriptions.py            # all families
    python scripts/add_moleculenet_descriptions.py --datasets toxcast,muv
"""

from __future__ import annotations

import argparse
import json

import _prepare_common as common
import moleculenet_descriptions as desc

SOURCE = "moleculenet"
TASK = "classification"


def _families() -> list[str]:
    base = common.PKG_DATA_ROOT / SOURCE / TASK
    return (
        sorted(p.name for p in base.iterdir() if (p / "metadata.json").exists())
        if base.exists()
        else []
    )


def _update_one(path, descriptions: dict[str, dict]) -> None:
    with open(path) as f:
        meta = json.load(f)
    for col, block in meta.get("columns", {}).items():
        d = descriptions.get(col, {})
        block["description"] = d.get("description")
        block["description_source"] = d.get("description_source")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def process(family: str) -> None:
    pkg = common.PKG_DATA_ROOT / SOURCE / TASK / family / "metadata.json"
    with open(pkg) as f:
        columns = list(json.load(f).get("columns", {}))
    descriptions = desc.describe_columns(family, columns)

    n_desc = sum(1 for c in columns if descriptions.get(c, {}).get("description"))
    for path in (pkg, common.DATA_ROOT / SOURCE / TASK / family / "metadata.json"):
        if path.exists():
            _update_one(path, descriptions)
    blank = len(columns) - n_desc
    note = f" ({blank} without a description)" if blank else ""
    print(f"  [{family}] described {n_desc}/{len(columns)} columns{note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add per-column descriptions to MoleculeNet metadata."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated families (default: all).",
    )
    args = parser.parse_args()

    families = _families()
    if args.datasets:
        want = {k.strip() for k in args.datasets.split(",") if k.strip()}
        families = [f for f in families if f in want]
    if not families:
        parser.error("no MoleculeNet families found (run prepare_moleculenet.py first)")

    for family in families:
        process(family)
    print(
        "\nDone. Re-bundled metadata carries column descriptions; re-upload data/ copies if publishing."
    )


if __name__ == "__main__":
    main()
