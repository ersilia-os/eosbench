"""Prepare TDC (Therapeutics Data Commons) single-input SMILES classification datasets.

Auto-discovers the single-prediction ADME, Tox and HTS datasets via PyTDC, keeps the
**binary-classification** ones (regression datasets are skipped automatically, each
logged with a reason), and writes one eosbench dataset *family* per dataset — each a
1-column family, since these TDC tasks are single-endpoint. Sizes span ~880 molecules
(SARS-CoV-2 assays) to ~340k (Butkiewicz HTS bioassays), giving benchmarks of varied scale.

Like the Polaris source (and unlike MoleculeNet, where we compute the scaffold split),
we honour **TDC's own scaffold split** as the holdout (`scaffold_split_method:
"tdc-scaffold"`) and add a random K-fold CV alongside. Features (morgan + rdkit) and a
RandomForest baseline are computed by ``scripts/_prepare_common.py``.

IMPORTANT — leaderboard references PREFER **Polaris** over TDC's own leaderboard (which can
contain errors). Published best-model scores are read from the curated
``scripts/tdcommons_leaderboard.json``; each entry's ``provider`` records where it came from
(``polaris`` primary, ``tdc`` fallback where Polaris has no entry). In practice TDC's only
single-molecule leaderboard IS the ADMET Benchmark Group, which Polaris mirrors, so there is
currently nothing to fall back to — datasets outside that group stay blank.

Usage::

    pip install -e ".[prepare]"
    python scripts/prepare_tdcommons.py                       # all binary ADMET datasets
    python scripts/prepare_tdcommons.py --datasets ames,herg  # specific datasets
    python scripts/prepare_tdcommons.py --limit 5 --no_baseline

Files are written locally only (data/ and src/eosbench/_data/); distribution is manual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _prepare_common as common

SOURCE = "tdcommons"
TASK = "classification"
HOLDOUT_METHOD = "tdc-scaffold"
LEADERBOARD_JSON = Path(__file__).resolve().parent / "tdcommons_leaderboard.json"

# TDC single_pred groups of single-input SMILES datasets we mine for binary classification:
# ADME, Tox (ADMET properties) and HTS (high-throughput screening bioassays). Regression
# datasets within these groups are skipped automatically. Other single_pred groups are
# excluded on purpose: QM/Yields are regression, and Epitope/Paratope/Develop/CRISPROutcome
# take protein/sequence inputs rather than SMILES.
GROUPS = ("ADME", "Tox", "HTS")


def load_leaderboard() -> dict:
    """Polaris-sourced leaderboard references, keyed by dataset slug."""
    with open(LEADERBOARD_JSON) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def discover() -> list[tuple[str, str]]:
    """Return ``(group, name)`` for every ADME, Tox and HTS single-pred dataset on TDC."""
    from tdc.utils import retrieve_dataset_names

    pairs: list[tuple[str, str]] = []
    for group in GROUPS:
        for name in retrieve_dataset_names(group):
            pairs.append((group, name))
    return pairs


def _group_class(group: str):
    from tdc.single_pred import ADME, HTS, Tox

    return {"ADME": ADME, "Tox": Tox, "HTS": HTS}[group]


def prepare_one(
    group: str, name: str, leaderboard: dict, n_folds: int, seed: int,
    compute_baseline: bool = True,
) -> int:
    """Prepare ONE dataset. Returns 1 if written, 0 if skipped (reason logged)."""
    cls = _group_class(group)
    try:
        d = cls(name=name)
    except Exception as e:  # noqa: BLE001 - e.g. multi-label datasets needing label_name
        print(f"  [{name}] skipped: load failed ({e})")
        return 0

    df = d.get_data()
    if not {"Drug", "Y", "Drug_ID"}.issubset(df.columns):
        print(f"  [{name}] skipped: unexpected columns {list(df.columns)}")
        return 0

    base = df[df["Drug"].notna()].reset_index(drop=True)
    coerced = common.coerce_binary(base["Y"])
    if coerced is None:
        print(f"  [{name}] skipped: not binary classification (likely regression)")
        return 0

    # Honour TDC's scaffold split as the holdout: mark test rows, everything else train.
    try:
        sp = d.get_split(method="scaffold", seed=seed)
        test_ids = set(sp["test"]["Drug_ID"])
    except Exception as e:  # noqa: BLE001
        print(f"  [{name}] skipped: scaffold split failed ({e})")
        return 0
    holdout = np.where(base["Drug_ID"].isin(test_ids), "test", "train").astype(object)

    label_df = pd.DataFrame({name: coerced.to_numpy()})
    common.prepare_family(
        source=SOURCE,
        family=name,
        smiles=base["Drug"].tolist(),
        label_df=label_df,
        task=TASK,
        n_folds=n_folds,
        seed=seed,
        leaderboard=leaderboard.get(name),
        compute_baseline=compute_baseline,
        holdout=holdout,
        holdout_method=HOLDOUT_METHOD,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TDC ADMET classification datasets for eosbench.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated TDC dataset names (e.g. ames,herg). Default: auto-discover all.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered datasets.")
    parser.add_argument("--n_folds", type=int, default=5, help="Random CV folds (default 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the RandomForest baseline; metrics are recorded as null (much faster).",
    )
    args = parser.parse_args()

    leaderboard = load_leaderboard()

    if args.datasets.strip():
        wanted = {n.strip() for n in args.datasets.split(",") if n.strip()}
        pairs = [(g, n) for (g, n) in discover() if n in wanted]
        missing = wanted - {n for _, n in pairs}
        if missing:
            parser.error(f"unknown TDC dataset(s): {', '.join(sorted(missing))}")
    else:
        print("Discovering ADME + Tox + HTS datasets on TDC...")
        pairs = discover()
        print(f"  found {len(pairs)} candidate dataset(s)")
    if args.limit is not None:
        pairs = pairs[: args.limit]

    total = 0
    for group, name in pairs:
        print(f"\n== {name} ({group}) ==")
        total += prepare_one(
            group, name, leaderboard, args.n_folds, args.seed,
            compute_baseline=not args.no_baseline,
        )

    print(f"\nDone: wrote {total} family(ies) from {len(pairs)} candidate(s) considered.")
    common.print_upload_hint()


if __name__ == "__main__":
    main()
