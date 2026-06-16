"""Prepare MoleculeNet binary-classification datasets for eosbench.

Downloads raw CSVs from the public DeepChem S3 mirror and writes one dataset *family* per
set (see scripts/_prepare_common.py). A family shares molecules, features, and a single
conserved train/test split across all its tasks; multi-task sets (Tox21, ClinTox, …) keep
each endpoint as a column with NaN where unlabeled, single-task sets are 1-task families.
Featurizes (morgan + rdkit), builds random K-fold and scaffold splits, computes a
RandomForest baseline, and attaches leaderboard references.

Usage::

    pip install -e ".[prepare]"
    python scripts/prepare_moleculenet.py                 # default subset
    python scripts/prepare_moleculenet.py --datasets bbbp,tox21
    python scripts/prepare_moleculenet.py --datasets all --n_folds 5 --seed 42

Files are written locally only (data/ and src/eosbench/_data/); distribution is manual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _prepare_common as common

SOURCE = "moleculenet"
TASK = "classification"
MIRROR = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
RAW_DIR = common.DATA_ROOT / "_raw" / "moleculenet"
LEADERBOARD_JSON = Path(__file__).resolve().parent / "moleculenet_leaderboard.json"

# Registry of MoleculeNet binary-classification sets. Each set becomes one family.
#   file        : filename on the DeepChem mirror
#   smiles_col  : SMILES column
#   single      : the single label column (single-column sets only)
#   columns     : explicit list of label columns (multi-column sets); None = auto-detect
REGISTRY: dict[str, dict] = {
    # --- single-column families ---
    "bbbp": {"file": "BBBP.csv", "smiles_col": "smiles", "single": "p_np"},
    "bace": {"file": "bace.csv", "smiles_col": "mol", "single": "Class"},
    "hiv":  {"file": "HIV.csv", "smiles_col": "smiles", "single": "HIV_active"},
    # --- multi-column families ---
    "clintox": {
        "file": "clintox.csv.gz", "smiles_col": "smiles",
        "columns": ["FDA_APPROVED", "CT_TOX"],
    },
    "tox21": {
        "file": "tox21.csv.gz", "smiles_col": "smiles",
        "columns": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
    },
    "sider": {"file": "sider.csv.gz", "smiles_col": "smiles", "columns": None},
    "muv":   {"file": "muv.csv.gz", "smiles_col": "smiles", "columns": None},
    "toxcast": {"file": "toxcast_data.csv.gz", "smiles_col": "smiles", "columns": None},
}

# Sensible default subset (excludes the very large muv/toxcast sweeps).
DEFAULT_SETS = ["bbbp", "bace", "hiv", "clintox", "tox21", "sider"]


def download(file: str) -> Path:
    """Download a raw MoleculeNet CSV from the DeepChem mirror, cached under data/_raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / file
    if dest.exists():
        return dest
    import urllib.request

    url = f"{MIRROR}/{file}"
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def label_columns(key: str, df: pd.DataFrame, spec: dict) -> list[str]:
    """Resolve the label columns for a set (auto-detect when 'columns' is None)."""
    if "single" in spec:
        return [spec["single"]]
    if spec.get("columns"):
        return spec["columns"]
    # Auto-detect: every column that isn't smiles/id is a label column.
    skip = {spec["smiles_col"], "mol_id", "smiles", "Index", "index"}
    return [c for c in df.columns if c not in skip]


def scrape_leaderboard() -> dict:
    """Best-effort fetch of moleculenet.org leaderboard numbers.

    moleculenet.org exposes no stable structured endpoint, so this is intentionally
    conservative: any failure returns ``{}`` and the caller falls back to the curated
    JSON. Kept as a hook so it can be fleshed out if/when a parseable source appears.
    """
    try:
        import urllib.request

        with urllib.request.urlopen("https://moleculenet.org/", timeout=10) as resp:
            resp.read()  # reachable, but no structured metrics to parse
    except Exception as e:  # noqa: BLE001
        print(f"  leaderboard scrape unavailable ({e}); using curated values")
        return {}
    print("  leaderboard scrape returned no structured data; using curated values")
    return {}


def load_leaderboard() -> dict:
    scraped = scrape_leaderboard()
    with open(LEADERBOARD_JSON) as f:
        curated = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    curated.update(scraped)  # scraped values take precedence when present
    return curated


def prepare_set(
    key: str, spec: dict, leaderboard: dict, n_folds: int, seed: int,
    compute_baseline: bool = True,
) -> int:
    """Prepare ONE family (≥1 binary tasks) from a MoleculeNet set. Returns 1 if written."""
    path = download(spec["file"])
    df = pd.read_csv(path)
    smiles_col = spec["smiles_col"]
    is_single = "single" in spec

    base = df[df[smiles_col].notna()].reset_index(drop=True)
    label_df = pd.DataFrame(index=base.index)
    for col in label_columns(key, df, spec):
        coerced = common.coerce_binary(base[col])
        if coerced is None:
            print(f"  [{key}/{col}] skipped task: not binary or single-class")
            continue
        # Single-task families name their sole task after the family for a friendly default.
        label_df[key if is_single else col] = coerced.to_numpy()

    if label_df.shape[1] == 0:
        print(f"  [{key}] skipped: no usable binary tasks")
        return 0

    common.prepare_family(
        source=SOURCE,
        family=key,
        smiles=base[smiles_col].tolist(),
        label_df=label_df,
        task=TASK,
        n_folds=n_folds,
        seed=seed,
        leaderboard=leaderboard.get(key),
        compute_baseline=compute_baseline,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MoleculeNet datasets for eosbench.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_SETS),
        help=f"Comma-separated set keys, or 'all'. Known: {', '.join(REGISTRY)}. "
        f"Default: {', '.join(DEFAULT_SETS)}.",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Random CV folds (default 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the per-task RandomForest baseline (much faster for large families "
        "like ToxCast/MUV); metrics are recorded as null.",
    )
    args = parser.parse_args()

    if args.datasets.strip() == "all":
        keys = list(REGISTRY)
    else:
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        parser.error(f"unknown dataset set(s): {unknown}. Known: {list(REGISTRY)}")

    leaderboard = load_leaderboard()

    total = 0
    for key in keys:
        print(f"\n== {key} ==")
        total += prepare_set(
            key, REGISTRY[key], leaderboard, args.n_folds, args.seed,
            compute_baseline=not args.no_baseline,
        )

    print(f"\nDone: wrote {total} family(ies).")
    common.print_upload_hint()


if __name__ == "__main__":
    main()
