"""Prepare TDC (Therapeutics Data Commons) single-input SMILES classification datasets.

Auto-discovers the single-prediction ADME, Tox and HTS datasets via PyTDC, keeps the
**binary-classification** ones (regression datasets are skipped automatically, each
logged with a reason), and writes one eosbench dataset *family* per dataset. Most are
1-column families (single-endpoint); a handful (tox21, toxcast, herg_central) are
multi-label on TDC and become multi-column families via `prepare_multi_label` -- TDC
serves each label as a separate call with its own molecule subset, so those are aligned
by Drug_ID into one conserved family rather than read off a single flat table. Sizes span
~880 molecules (SARS-CoV-2 assays) to ~340k (Butkiewicz HTS bioassays, herg_central).

Like the Polaris source (and unlike MoleculeNet, where we compute the scaffold split),
single-label datasets honour **TDC's own scaffold split** as the holdout
(`scaffold_split_method: "tdc-scaffold"`), generated with the fixed ``TDC_SCAFFOLD_SEED``
(see its definition below for why that reproduces the ADMET Benchmark Group's leaderboard
test set exactly); multi-label families have no such split available (TDC's per-label
splits don't cover the assembled union) and get eosbench's own computed Murcko scaffold
split instead, like MoleculeNet's multi-column families. A random K-fold CV is added
alongside either way. Features (morgan + rdkit) and a RandomForest baseline are computed
by ``scripts/_prepare_common.py``.

IMPORTANT — leaderboard references PREFER **Polaris** over TDC's own leaderboard (which can
contain errors). Published best-model scores are read from the curated
``scripts/tdcommons_leaderboard.json``; each entry's ``provider`` records where it came from
(``polaris`` primary, ``tdc`` fallback where Polaris has no entry). In practice TDC's only
single-molecule leaderboard IS the ADMET Benchmark Group, which Polaris mirrors, so there is
currently nothing to fall back to — datasets outside that group stay blank.

Usage::

    pip install -e ".[prepare-tdcommons]"
    python scripts/prepare_tdcommons.py                       # all binary ADMET datasets
    python scripts/prepare_tdcommons.py --datasets ames,herg  # specific datasets
    python scripts/prepare_tdcommons.py --limit 5 --no_baseline

Files are written locally only (data/ and src/eosbench/_data/); distribution is manual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _prepare_common as common
import numpy as np
import pandas as pd

SOURCE = "tdcommons"
TASK = "classification"
HOLDOUT_METHOD = "tdc-scaffold"
# TDC's ADMET Benchmark Group ships a *frozen* test.csv per dataset for its leaderboard;
# `get_split()` doesn't read that file -- it independently recomputes a scaffold split with
# whatever seed it's given. Verified empirically (via tdc.benchmark_group.admet_group()) that
# calling get_split(method="scaffold", seed=42) reproduces that frozen test set byte-for-byte
# (exact Drug_ID match) for every ADMET Benchmark Group member checked -- 42 is the seed TDC
# itself used to generate those files, not an arbitrary choice. Kept as its own constant,
# independent of --seed (which also drives the random CV folds and the RF random_state), so
# re-running with a different --seed for those unrelated reasons can never silently break
# leaderboard comparability.
TDC_SCAFFOLD_SEED = 42
LEADERBOARD_JSON = Path(__file__).resolve().parent / "tdcommons_leaderboard.json"
# PyTDC's dataset loaders default to path="./data" for their own raw-download cache.
# common.DATA_ROOT is also "./data" (relative to the repo root) -- the exact directory
# prepare_family() writes eosbench's own output into and that later gets published wholesale
# (e.g. `eosvc upload --path data/`). Left at the default, TDC's raw *.tab cache lands
# directly inside that tree. Redirect it to a sibling, un-published cache dir instead.
TDC_RAW_CACHE = common.REPO_ROOT / ".tdc_raw_cache"

# TDC single_pred groups of single-input SMILES datasets we mine for binary classification:
# ADME, Tox (ADMET properties) and HTS (high-throughput screening bioassays). Regression
# datasets within these groups are skipped automatically. Other single_pred groups are
# excluded on purpose: QM/Yields are regression, and Epitope/Paratope/Develop/CRISPROutcome
# take protein/sequence inputs rather than SMILES.
GROUPS = ("ADME", "Tox", "HTS")


def load_leaderboard() -> dict:
    """Curated leaderboard references, keyed by dataset slug.

    Returns
    -------
    dict
    """
    with open(LEADERBOARD_JSON) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def discover() -> list[tuple[str, str]]:
    """Return ``(group, name)`` for every ADME, Tox and HTS single-pred dataset on TDC.

    Returns
    -------
    list of tuple of (str, str)
    """
    from tdc.utils import retrieve_dataset_names

    pairs: list[tuple[str, str]] = []
    for group in GROUPS:
        for name in retrieve_dataset_names(group):
            pairs.append((group, name))
    return pairs


def _group_class(group: str):
    from tdc.single_pred import ADME, HTS, Tox

    return {"ADME": ADME, "Tox": Tox, "HTS": HTS}[group]


def _label_names(name: str) -> list[str] | None:
    """Label names for a multi-label TDC dataset (e.g. tox21's 12 endpoints), or None for
    a single-label one. TDC raises for the latter, so that's the signal we key off of --
    more robust than matching the "please select a label name" text raised deeper inside
    a dataset-loader call."""
    from tdc.utils import retrieve_label_name_list

    try:
        return retrieve_label_name_list(name)
    except Exception:  # noqa: BLE001
        return None


def prepare_multi_label(
    group: str,
    name: str,
    labels: list[str],
    leaderboard: dict,
    n_folds: int,
    seed: int,
    compute_baseline: bool = True,
) -> int:
    """Prepare a multi-label TDC dataset (e.g. tox21, toxcast, herg_central) as one family.

    Unlike MoleculeNet's multi-column sets (one flat CSV, every label pre-aligned by row),
    TDC serves each label of a multi-label dataset as a *separate* call, and different
    labels can cover different molecule sets (e.g. tox21's NR-AR: 7265 rows vs its
    NR-AhR: 6549, only 6460 overlapping). So this aligns labels itself: a Drug_ID -> Drug
    (SMILES) map plus one Drug_ID -> y map per binary-coercible label, then builds the
    family over the *union* of Drug_IDs (NaN where a molecule wasn't in a given label's
    set) -- the same "conserved family" shape every other source produces.

    No family-level holdout is passed to `prepare_family`: TDC has no split that covers
    the whole union (each label's own `get_split()` only spans that label's own molecule
    subset), so this deliberately falls back to `prepare_family`'s own internal Murcko
    scaffold split -- exactly what `prepare_moleculenet.py` already does for all of *its*
    multi-column families (tox21, toxcast, clintox, sider, muv).

    Parameters
    ----------
    group : str
        TDC single_pred group, e.g. "Tox".
    name : str
        TDC dataset name, e.g. "tox21".
    labels : list of str
        Label names to fetch and align.
    leaderboard : dict
        Curated leaderboard references, as returned by :func:`load_leaderboard`.
    n_folds : int
        Number of random CV folds.
    seed : int
        Random seed for the CV folds and the RandomForest baseline.
    compute_baseline : bool, default True
        Whether to train the RandomForest baseline.

    Returns
    -------
    int
        1 if the family was written, 0 if skipped.
    """
    cls = _group_class(group)
    drug_map: dict[str, str] = {}
    label_series: dict[str, dict[str, float]] = {}

    for label in labels:
        try:
            df = cls(name=name, label_name=label, path=str(TDC_RAW_CACHE)).get_data()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}/{label}] skipped: load failed ({e})")
            continue
        if not {"Drug", "Y", "Drug_ID"}.issubset(df.columns):
            print(f"  [{name}/{label}] skipped: unexpected columns {list(df.columns)}")
            continue
        coerced = common.coerce_binary(df["Y"])
        if coerced is None:
            print(
                f"  [{name}/{label}] skipped: not binary classification (likely regression)"
            )
            continue
        for drug_id, drug in zip(df["Drug_ID"], df["Drug"]):
            drug_map.setdefault(drug_id, drug)
        label_series[label] = dict(zip(df["Drug_ID"], coerced))

    if not label_series:
        print(f"  [{name}] skipped: no usable binary labels")
        return 0

    drug_ids = sorted(drug_map)  # deterministic order
    smiles = [drug_map[d] for d in drug_ids]
    label_df = pd.DataFrame(
        {
            label: [series.get(d) for d in drug_ids]
            for label, series in label_series.items()
        }
    )

    common.prepare_family(
        source=SOURCE,
        family=name,
        smiles=smiles,
        label_df=label_df,
        task=TASK,
        n_folds=n_folds,
        seed=seed,
        leaderboard=leaderboard.get(name),
        compute_baseline=compute_baseline,
    )
    return 1


def prepare_one(
    group: str,
    name: str,
    leaderboard: dict,
    n_folds: int,
    seed: int,
    compute_baseline: bool = True,
) -> int:
    """Prepare ONE dataset.

    Parameters
    ----------
    group : str
        TDC single_pred group, e.g. "Tox".
    name : str
        TDC dataset name, e.g. "ames".
    leaderboard : dict
        Curated leaderboard references, as returned by :func:`load_leaderboard`.
    n_folds : int
        Number of random CV folds.
    seed : int
        Random seed for the CV folds and the RandomForest baseline.
    compute_baseline : bool, default True
        Whether to train the RandomForest baseline.

    Returns
    -------
    int
        1 if written, 0 if skipped (reason logged).
    """
    labels = _label_names(name)
    if labels is not None:
        return prepare_multi_label(
            group, name, labels, leaderboard, n_folds, seed, compute_baseline
        )

    cls = _group_class(group)
    try:
        d = cls(name=name, path=str(TDC_RAW_CACHE))
    except Exception as e:  # noqa: BLE001
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

    # Guard the TDC_SCAFFOLD_SEED assumption above: we always request a *scaffold* split
    # below, which only reproduces the ADMET Benchmark Group's leaderboard test set when
    # that dataset's official split really is "scaffold" (true for all 22 members today,
    # per tdc.metadata.bm_split_names["admet_group"] -- but unchecked, so a future TDC
    # dataset using a different split would otherwise silently get an incomparable
    # scaffold_auroc next to a "polaris" leaderboard reference). Hard stop rather than
    # warn-and-continue: this would invalidate leaderboard comparability, not just this
    # one dataset's numbers.
    lb_entry = leaderboard.get(name, {})
    if lb_entry.get("provider") == "polaris" and lb_entry.get("split") not in (
        None,
        "scaffold",
    ):
        sys.exit(
            f"[{name}] leaderboard split is {lb_entry['split']!r} (provider=polaris), but "
            "prepare_one() always computes a scaffold holdout -- scaffold_auroc would not "
            "be comparable to the leaderboard reference. Fix TDC_SCAFFOLD_SEED/get_split() "
            "handling for this dataset before proceeding."
        )

    # Honour TDC's scaffold split as the holdout: mark test rows, everything else train.
    # Uses TDC_SCAFFOLD_SEED (not the CV/RF `seed` param) -- see its definition above.
    try:
        sp = d.get_split(method="scaffold", seed=TDC_SCAFFOLD_SEED)
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
    """Entry point: prepare the requested (or all discovered) TDC ADMET datasets."""
    parser = argparse.ArgumentParser(
        description="Prepare TDC ADMET classification datasets for eosbench."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated TDC dataset names (e.g. ames,herg). Default: auto-discover all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered datasets.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Random CV folds (default 5)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default 42)."
    )
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
            group,
            name,
            leaderboard,
            args.n_folds,
            args.seed,
            compute_baseline=not args.no_baseline,
        )

    print(
        f"\nDone: wrote {total} family(ies) from {len(pairs)} candidate(s) considered."
    )
    common.print_upload_hint()


if __name__ == "__main__":
    main()
