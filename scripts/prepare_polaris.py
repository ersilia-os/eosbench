"""Prepare Polaris Hub single-input classification benchmarks for eosbench.

Auto-discovers benchmarks on the Polaris Hub, keeps the **single-input, binary-classification**
ones, and writes one eosbench dataset *family* per benchmark (see scripts/_prepare_common.py).
Single-target benchmarks become 1-task families; single-input multi-target benchmarks become
multi-task families (one column per target, NaN where unlabeled).

Unlike MoleculeNet (where we compute a scaffold split), Polaris ships an *official* train/test
split, so we honour it as the holdout and additionally build a random K-fold CV. Polaris hides
the test labels only behind its split API; the labels live in ``benchmark.dataset.table``, which
we read directly to reconstruct the full labelled data.

Usage::

    pip install -e ".[prepare-polaris]"
    polaris login                                  # one-time, cached token
    python scripts/prepare_polaris.py                          # all qualifying benchmarks
    python scripts/prepare_polaris.py --datasets owner/slug    # a specific benchmark
    python scripts/prepare_polaris.py --limit 5 --no_baseline  # quick pass

Files are written locally only (data/ and src/eosbench/_data/); distribution is manual.
"""

from __future__ import annotations

import argparse
import re

import _prepare_common as common
import numpy as np

SOURCE = "polaris"
TASK = "classification"


def slugify(benchmark_id: str) -> str:
    """Turn an ``owner/benchmark-name`` id into a clean dataset directory name.

    Parameters
    ----------
    benchmark_id : str
        Benchmark id as ``"owner/slug"``.

    Returns
    -------
    str
    """
    name = benchmark_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def list_benchmark_ids() -> list[str]:
    """Return all benchmark ids on the hub as ``owner/slug`` strings (paginated).

    Returns
    -------
    list of str
    """
    from polaris.hub.client import PolarisHubClient

    ids: list[str] = []
    page = 100
    with PolarisHubClient() as client:
        offset = 0
        while True:
            batch = client.list_benchmarks(limit=page, offset=offset)
            if not batch:
                break
            ids.extend(str(b) for b in batch)
            if len(batch) < page:
                break
            offset += page
    return ids


def _is_classification(target_type) -> bool:
    val = getattr(target_type, "value", target_type)
    return str(val).lower() == "classification"


def _split_rows(benchmark) -> tuple[list[int], np.ndarray]:
    """Return the dataset rows the benchmark uses and their 'train'/'test' holdout.

    A benchmark's split indexes into its dataset table, which may hold *more* rows than the
    split references — so we restrict to ``train ∪ test`` rather than assuming every row is
    used. Returns ``(row_indices, holdout)`` aligned to each other.
    """
    train, test = benchmark.split
    train_idx = [int(i) for i in train]
    test_idx: set[int] = set()
    if isinstance(test, dict):
        for idx in test.values():
            test_idx.update(int(i) for i in idx)
    else:
        test_idx.update(int(i) for i in test)
    rows = sorted(set(train_idx) | test_idx)
    holdout = np.array(
        ["test" if i in test_idx else "train" for i in rows], dtype=object
    )
    return rows, holdout


def prepare_benchmark(
    benchmark_id: str, n_folds: int, seed: int, compute_baseline: bool = True
) -> int:
    """Prepare ONE benchmark.

    Parameters
    ----------
    benchmark_id : str
        Benchmark id as ``"owner/slug"``, e.g. "tdcommons/ames".
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
    import polaris as po

    try:
        bench = po.load_benchmark(benchmark_id)
    except Exception as e:  # noqa: BLE001
        print(f"  [{benchmark_id}] skipped: load failed ({e})")
        return 0

    input_cols = sorted(bench.input_cols)
    if len(input_cols) != 1:
        print(f"  [{benchmark_id}] skipped: multi-input ({input_cols})")
        return 0
    input_col = input_cols[0]

    target_cols = sorted(bench.target_cols)
    target_types = dict(getattr(bench, "target_types", {}) or {})
    non_clf = [
        c
        for c in target_cols
        if c in target_types and not _is_classification(target_types[c])
    ]
    if non_clf:
        print(f"  [{benchmark_id}] skipped: non-classification targets {non_clf}")
        return 0

    try:
        table = bench.dataset.table.reset_index(drop=True)
        rows, holdout = _split_rows(bench)
        df = table.iloc[rows].reset_index(drop=True)
    except Exception as e:  # noqa: BLE001 - e.g. V2 zarr-backed datasets without a flat table
        print(f"  [{benchmark_id}] skipped: could not recover table/split ({e})")
        return 0

    if input_col not in df.columns:
        print(f"  [{benchmark_id}] skipped: input column '{input_col}' not in table")
        return 0

    slug = slugify(benchmark_id)
    is_single = len(target_cols) == 1
    import pandas as pd

    label_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        if col not in df.columns:
            print(f"  [{benchmark_id}/{col}] skipped task: column not in table")
            continue
        coerced = common.coerce_binary(df[col])
        if coerced is None:
            print(f"  [{benchmark_id}/{col}] skipped task: not binary or single-class")
            continue
        # Single-task families name their sole task after the family for a friendly default.
        label_df[slug if is_single else col] = coerced.to_numpy()

    if label_df.shape[1] == 0:
        print(f"  [{benchmark_id}] skipped: no usable binary classification tasks")
        return 0

    common.prepare_family(
        source=SOURCE,
        family=slug,
        smiles=df[input_col].tolist(),
        label_df=label_df,
        task=TASK,
        n_folds=n_folds,
        seed=seed,
        leaderboard=None,
        compute_baseline=compute_baseline,
        holdout=holdout,
        holdout_method="polaris",
    )
    return 1


def main() -> None:
    """Entry point: prepare the requested (or auto-discovered) Polaris benchmarks."""
    parser = argparse.ArgumentParser(
        description="Prepare Polaris benchmarks for eosbench."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated benchmark ids (owner/slug). Default: auto-discover all.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N benchmarks."
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
        help="Skip the per-task RandomForest baseline; metrics are recorded as null.",
    )
    args = parser.parse_args()

    if args.datasets.strip():
        ids = [b.strip() for b in args.datasets.split(",") if b.strip()]
    else:
        print("Discovering benchmarks on the Polaris Hub...")
        ids = list_benchmark_ids()
        print(f"  found {len(ids)} benchmark(s)")
    if args.limit is not None:
        ids = ids[: args.limit]

    total = 0
    for bid in ids:
        print(f"\n== {bid} ==")
        total += prepare_benchmark(
            bid, args.n_folds, args.seed, compute_baseline=not args.no_baseline
        )

    print(f"\nDone: wrote {total} family(ies) from {len(ids)} benchmark(s) considered.")
    common.print_upload_hint()


if __name__ == "__main__":
    main()
