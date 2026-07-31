from collections.abc import Iterator
import hashlib
import shutil
import json
import os
from importlib import resources
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd

from .utils.logging import logger

S3_BASE = "https://eosvc-public.s3.amazonaws.com/eosbench/data"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "eosbench")

# Supported featurizations and their output dimensionalities.
FEATURIZATIONS = {
    "morgan": 2048,  # Morgan fingerprints (counts), int64
    "rdkit": 217,  # RDKit physicochemical descriptors, float64
}


def _cache_path(source: str, task: str, dataset: str, filename: str) -> str:
    path = os.path.join(CACHE_DIR, source, task, dataset, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _rmdir_if_empty_chain(path: Path, stop: Path) -> None:
    """Remove ``path`` and now-empty parent directories, without touching ``stop`` itself.

    Used to clean up the directory chain a failed fetch would otherwise leave behind
    (created eagerly so a download has somewhere to write to). A directory holding any
    file — including one left by a partially successful fetch — is kept as-is.
    """
    path = Path(path)
    stop = Path(stop)
    while path != stop and path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        path = path.parent


def _head_ok(url: str, timeout: float = 15.0) -> bool:
    """True if a HEAD request to ``url`` returns HTTP 200 (object exists on S3)."""
    try:
        with urlopen(Request(url, method="HEAD"), timeout=timeout) as resp:
            return resp.status == 200
    except (URLError, OSError, ValueError):
        return False


def check_availability(source: str, task: str, dataset: str, timeout: float = 15.0) -> bool:
    """True if this dataset family is present on the public S3 bucket right now.

    Performs a live network HEAD request against the family's ``data.csv`` — the same
    check the catalog website (``eosbench catalog`` docs, ``site/index.html``) runs. This
    is a network call: use it sparingly (e.g. behind an opt-in CLI flag), not on every
    catalog load.
    """
    url = f"{S3_BASE}/{source}/{task}/{dataset}/data.csv"
    return _head_ok(url, timeout=timeout)


def _download_to(
    source: str,
    task: str,
    dataset: str,
    filename: str,
    dest: str | os.PathLike,
) -> str:
    dest = str(dest)
    if os.path.exists(dest):
        logger.debug(f"Cache hit: {dest}")
    else:
        url = f"{S3_BASE}/{source}/{task}/{dataset}/{filename}"
        logger.info(f"Downloading {filename} for {source}/{dataset}...")
        try:
            urlretrieve(url, dest)
        except URLError as e:
            if getattr(e, "code", None) == 404:
                raise RuntimeError(
                    f"{source}/{task}/{dataset} is not available on S3 yet (no {filename} "
                    "found). Run `eosbench catalog` to check current availability."
                ) from e
            raise RuntimeError(f"Failed to download {url}: {e}") from e
        logger.success(f"Saved {filename} to cache")
    return dest


def _copy_from_dir(
    source: str,
    task: str,
    dataset: str,
    filename: str,
    dest: str | os.PathLike,
    from_dir: str | os.PathLike,
) -> str:
    """Copy a dataset file from a local mirror directory instead of S3.

    ``from_dir`` is expected to follow the same ``{source}/{task}/{dataset}/{filename}``
    layout the prepare scripts write under ``data/``. Like :func:`_download_to`, an existing
    destination is a cache hit and is left untouched.
    """
    dest = str(dest)
    if os.path.exists(dest):
        logger.debug(f"Cache hit: {dest}")
        return dest
    src = Path(from_dir) / source / task / dataset / filename
    if not src.exists():
        raise FileNotFoundError(
            f"{filename} not found for {source}/{task}/{dataset} under --from_dir: {src}"
        )
    shutil.copy2(src, dest)
    logger.info(f"Copied {filename} from {src}")
    return dest


def _fetch(source: str, task: str, dataset: str, filename: str) -> str:
    dest = _cache_path(source, task, dataset, filename)
    try:
        return _download_to(source, task, dataset, filename, dest)
    except Exception:
        _rmdir_if_empty_chain(Path(dest).parent, Path(CACHE_DIR))
        raise


def _pkg_data_path(source: str, task: str, dataset: str, filename: str) -> str:
    return str(
        resources.files("eosbench") / "_data" / source / task / dataset / filename
    )


def _output_dataset_dir(
    output_dir: str | os.PathLike, source: str, task: str, dataset: str
) -> Path:
    return Path(output_dir) / source / task / dataset


class Splits:
    """Train/test index splits.

    Backs two split strategies behind a single interface:

    - **Cross-validation** (``from_folds``): an integer fold per sample yields one
      leave-one-fold-out ``(train_idxs, test_idxs)`` pair per fold.
    - **Holdout** (``from_holdout``): a per-sample ``"train"``/``"test"`` label yields a
      single ``(train_idxs, test_idxs)`` pair (``len == 1``).

    Supports iteration and direct indexing::

        for train_idxs, test_idxs in dataset.split():
            ...

        train_idxs, test_idxs = dataset.split[0]
    """

    def __init__(self, splits: list[tuple[np.ndarray, np.ndarray]]):
        self._splits = splits

    @classmethod
    def from_folds(cls, folds: np.ndarray) -> "Splits":
        """Build leave-one-fold-out CV splits from an integer fold array."""
        unique = sorted(set(folds.tolist()))
        return cls([(np.where(folds != k)[0], np.where(folds == k)[0]) for k in unique])

    @classmethod
    def from_holdout(cls, labels: np.ndarray) -> "Splits":
        """Build a single train/test split from per-sample ``"train"``/``"test"`` labels."""
        labels = np.asarray(labels)
        return cls([(np.where(labels == "train")[0], np.where(labels == "test")[0])])

    def __call__(self):
        return iter(self._splits)

    def __getitem__(self, i):
        return self._splits[i]

    def __len__(self):
        return len(self._splits)

    def __iter__(self):
        return iter(self._splits)


class Dataset:
    """A benchmark dataset with features, labels, CV splits, and metadata.

    Attributes
    ----------
    X : list[str] or np.ndarray
        SMILES strings (featurization=None), or a descriptor matrix:
        (n, 2048) for "morgan", (n, 217) for "rdkit".
    y : np.ndarray
        Target values, shape (n,).
    split : Splits
        Train/test splits (CV folds or a single holdout, depending on the
        ``split`` argument to :func:`load_dataset`). Iterable and indexable::

            for train_idxs, test_idxs in dataset.split():
                ...
            train_idxs, test_idxs = dataset.split[0]

    metadata : dict
        Dataset metadata including baseline performance metrics.
    """

    def __init__(self, X, y: np.ndarray, split: Splits, metadata: dict):
        self.X = X
        self.y = y
        self.split = split
        self.metadata = metadata


class DatasetInfo:
    """Lightweight dataset descriptor — metadata only, no data downloaded.

    Attributes
    ----------
    source : str
    dataset : str
    task : str
    metadata : dict
    """

    def __init__(self, source: str, task: str, dataset: str):
        self.source = source
        self.task = task
        self.dataset = dataset
        meta_path = _pkg_data_path(source, task, dataset, "metadata.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)

    @property
    def id(self) -> str:
        """Deterministic short eosbench identifier for this dataset family."""
        return make_id(self.source, self.dataset)

    def column_id(self, column: str) -> str:
        """Deterministic short eosbench identifier for one label column."""
        return make_id(self.source, self.dataset, column)

    @property
    def columns(self) -> list[str]:
        """Label-column (endpoint) names. A single-element list for single-task datasets."""
        columns = self.metadata.get("columns")
        return list(columns) if columns else [self.dataset]

    def __repr__(self):
        return (
            f"DatasetInfo(source={self.source!r}, dataset={self.dataset!r}, "
            f"task={self.task!r}, n_columns={len(self.columns)})"
        )

    def load(
        self,
        featurization: str | None = "morgan",
        split: str = "random",
        column: str | None = None,
    ) -> "Dataset":
        """Download and return the full Dataset.

        Parameters
        ----------
        featurization : str or None
            One of "morgan", "rdkit", or None (SMILES).
        split : str
            "random" (K-fold CV) or "scaffold" (fixed train/test holdout).
        column : str or None
            For multi-column families, which label column to load (see :func:`load_dataset`).
        """
        return load_dataset(
            self.source,
            self.dataset,
            featurization,
            task=self.task,
            split=split,
            column=column,
        )


def _build_splits(
    folds_df: pd.DataFrame, split: str, mask: np.ndarray | None = None
) -> Splits:
    """Build a :class:`Splits` from a folds.csv frame for the requested strategy.

    Recognises the multi-column schema (``random_fold`` / ``scaffold_split``) and the
    legacy single ``fold`` column. When ``mask`` (a boolean array over the rows) is given,
    the split is restricted to the selected rows — used to view a family's conserved split
    through a single task's labeled rows.
    """
    if split == "random":
        if "random_fold" in folds_df.columns:
            col = "random_fold"
        elif "fold" in folds_df.columns:  # legacy datasets
            col = "fold"
        else:
            raise ValueError(
                "folds.csv has no 'random_fold' or 'fold' column for split='random'"
            )
        arr = folds_df[col].values.astype(int)
        return Splits.from_folds(arr if mask is None else arr[mask])
    elif split == "scaffold":
        if "scaffold_split" not in folds_df.columns:
            raise ValueError(
                "folds.csv has no 'scaffold_split' column; this dataset has no "
                "predefined scaffold split (try split='random')"
            )
        arr = folds_df["scaffold_split"].astype(str).values
        return Splits.from_holdout(arr if mask is None else arr[mask])
    raise ValueError(f"split must be 'random' or 'scaffold', got {split!r}")


def _resolve_column(metadata: dict, column: str | None) -> str | None:
    """Resolve the requested label column against a family's metadata.

    Returns the column name for family-format datasets, or ``None`` for legacy single-task
    datasets (no ``columns`` block). Raises if a multi-column family is loaded without
    ``column`` or with an unknown one.
    """
    columns = metadata.get("columns")
    if not columns:  # legacy single-task dataset
        return None
    names = list(columns)
    if column is None:
        if len(names) == 1:
            return names[0]
        raise ValueError(
            f"dataset {metadata.get('dataset')!r} has multiple columns; pass column=<one of "
            f"{names}>"
        )
    if column not in columns:
        raise ValueError(f"unknown column {column!r}; available columns: {names}")
    return column


def load_dataset(
    source: str,
    dataset: str,
    featurization: str | None = "morgan",
    task: str = "classification",
    split: str = "random",
    column: str | None = None,
) -> Dataset:
    """Load a benchmark dataset.

    Parameters
    ----------
    source : str
        Dataset source, e.g. "tdcommons" or "moleculenet".
    dataset : str
        Dataset (family) name, e.g. "ames", "tox21", or "bbbp".
    featurization : str or None
        One of "morgan", "rdkit", or None.
        If None, X is a list of SMILES strings.
        "morgan" → (n, 2048), "rdkit" → (n, 217).
    task : str
        "classification" or "regression" (default: "classification").
    split : str
        "random" for K-fold cross-validation (default) or "scaffold" for a fixed
        train/test holdout. "random" falls back to the legacy single ``fold`` column.
    column : str or None
        For multi-column families, which label column (endpoint) to load. ``None`` selects
        the sole column of a single-column family; for a multi-column family it raises with
        the column list. Ignored for legacy single-task datasets.

    Returns
    -------
    Dataset
    """
    if featurization is not None and featurization not in FEATURIZATIONS:
        raise ValueError(
            f"featurization must be None or one of {list(FEATURIZATIONS)}, got {featurization!r}"
        )

    logger.debug(
        f"Loading dataset {source}/{dataset} "
        f"(featurization={featurization}, split={split}, column={column})"
    )

    meta_path = _pkg_data_path(source, task, dataset, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    resolved = _resolve_column(metadata, column)

    csv_path = _fetch(source, task, dataset, "data.csv")
    df = pd.read_csv(csv_path)

    if resolved is not None:  # family format: select label column, mask labeled rows
        mask = df[resolved].notna().to_numpy()
        y = df.loc[mask, resolved].to_numpy()
        # Classification labels are integer-coded; regression targets stay float.
        y = y.astype(int) if task == "classification" else y.astype(float)
        smiles = df.loc[mask, "smiles"].tolist()
        metadata = {**metadata, "column": resolved, **metadata["columns"][resolved]}
    else:  # legacy single-task: activity/value column, all rows
        mask = None
        activity_col = "activity" if "activity" in df.columns else "value"
        y = df[activity_col].to_numpy()
        smiles = df["smiles"].tolist()

    if featurization is None:
        X = smiles
        logger.debug(f"X: {len(X)} SMILES strings")
    else:
        feats = np.load(_fetch(source, task, dataset, f"{featurization}.npy"))
        X = feats if mask is None else feats[mask]
        logger.debug(f"X: {X.shape} ({featurization})")

    folds_path = _fetch(source, task, dataset, "folds.csv")
    splits = _build_splits(pd.read_csv(folds_path), split, mask)

    logger.debug(f"y: {y.shape}, split={split} ({len(splits)} fold(s))")
    return Dataset(X, y, splits, metadata)


def list_sources() -> list[str]:
    """Return the available dataset sources by scanning bundled ``_data``."""
    base = resources.files("eosbench") / "_data"
    try:
        return sorted(entry.name for entry in base.iterdir() if entry.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        return []


def iter_datasets(
    source: str,
    task: str = "classification",
) -> Iterator[DatasetInfo]:
    """Iterate over available datasets for a source and task type.

    Yields lightweight :class:`DatasetInfo` objects with metadata only —
    no data is downloaded.

    Parameters
    ----------
    source : str
        e.g. "tdcommons" or "moleculenet".
    task : str
        "classification" or "regression" (default: "classification").

    Yields
    ------
    DatasetInfo
    """
    base = resources.files("eosbench") / "_data" / source / task
    try:
        entries = sorted(entry.name for entry in base.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    for name in entries:
        yield DatasetInfo(source, task, name)


def list_datasets(
    source: str | None = None, task: str = "classification"
) -> list[dict]:
    """List available datasets.

    Parameters
    ----------
    source : str or None
        Filter by source (e.g. "tdcommons"). If None, returns all.
    task : str
        "classification" or "regression" (default: "classification").

    Returns
    -------
    list of dicts with keys: source, dataset, task.
    """
    sources = list_sources() if source is None else [source]
    result = []
    for src in sources:
        for info in iter_datasets(src, task):
            result.append({"source": src, "dataset": info.dataset, "task": task})
    return sorted(result, key=lambda d: (d["source"], d["dataset"]))


def get_path(
    base_dir: str | os.PathLike,
    dataset_name: str,
    source: str,
    task: str,
) -> Path:
    """Return the path to a dataset directory within a base directory.

    Parameters
    ----------
    base_dir : str or PathLike
        Root folder (e.g. the ``output_dir`` passed to ``mirror_dataset``).
    dataset_name : str
        Dataset name, e.g. "ames".
    source : str
        e.g. "tdcommons" or "moleculenet".
    task : str
        "classification" or "regression".

    Returns
    -------
    Path
    """
    return Path(base_dir) / source / task / dataset_name


def list_columns(source: str, dataset: str, task: str = "classification") -> list[str]:
    """Return the label-column (endpoint) names of a dataset family.

    A single-element list (the dataset name) for single-task datasets.
    """
    return DatasetInfo(source, task, dataset).columns


def _ratio(n_tot, n_pos):
    return n_pos / n_tot if n_tot and n_pos is not None else None


def _median_int(values):
    """Median of a list of counts as an int, or None if there are no values."""
    vals = [v for v in values if v is not None]
    return int(round(float(np.median(vals)))) if vals else None


def _median_float(values):
    """Median of a list of floats, or None if there are no (non-None) values."""
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


# Per-task catalog metric contract. Each metric maps a display column to the
# metadata keys holding its value: (display name, collapsed family key, per-column key).
# Regression keys are forward-looking — no regression data is bundled yet; the prep
# pipeline must populate these names when it gains a regression baseline.
TASK_METRICS = {
    "classification": {
        "metrics": [
            ("auroc", "auroc_mean", "random_auroc_mean"),
            ("auprc", "aupr_mean", "random_aupr_mean"),
        ],
        "show_class_balance": True,  # include n_pos and ratio
    },
    "regression": {
        "metrics": [
            ("rmse", "rmse_mean", "random_rmse_mean"),
            ("r2", "r2_mean", "random_r2_mean"),
        ],
        "show_class_balance": False,
    },
}


def _task_metric_spec(task: str) -> dict:
    """Metric spec for a task type, falling back to classification."""
    return TASK_METRICS.get(task, TASK_METRICS["classification"])


def make_id(source: str, dataset: str, column: str | None = None) -> str:
    """Deterministic short eosbench identifier for a dataset family or label column.

    A stable hex handle derived purely from ``source``/``dataset`` (and ``column`` for an
    endpoint), so it is reproducible across regenerations and needs no registry. Family:
    ``make_id("tdcommons", "ames") -> "a3f9c2b1"``; column:
    ``make_id("moleculenet", "tox21", "NR-AR")`` gives a different code.
    """
    key = f"{source}/{dataset}" + (f"/{column}" if column is not None else "")
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def resolve_id(identifier: str) -> dict:
    """Resolve an eosbench id back to its dataset (and column, if it's a column id).

    Scans the bundled catalog for a family or label-column whose :func:`make_id` matches.
    Returns ``{"source", "dataset", "task", "column"}`` (``column`` is None for a family id).
    Raises ``KeyError`` if no dataset has that id.
    """
    for source in list_sources():
        for task in ("classification", "regression"):
            for info in iter_datasets(source, task):
                if info.id == identifier:
                    return {"source": source, "dataset": info.dataset, "task": task, "column": None}
                for col in info.columns:
                    if info.column_id(col) == identifier:
                        return {"source": source, "dataset": info.dataset, "task": task, "column": col}
    raise KeyError(f"no eosbench dataset or column has id {identifier!r}")


def catalog_columns(task: str = "classification", expand: bool = False) -> list[str]:
    """Column order of a ``get_catalog`` frame for the given task and view.

    ``task="all"`` returns the union of the classification and regression columns (both
    metric sets, plus the class-balance columns), so a combined catalog renders uniformly.
    """
    if task == "all":
        return [
            "id", "name", "source", "task", "column" if expand else "n_columns", "n_tot",
            "size", "n_pos", "auroc", "auprc", "ratio", "rmse", "r2", "skew",
            "leaderboard_score", "leaderboard_metric", "last_updated",
        ]
    spec = _task_metric_spec(task)
    cols = ["id", "name", "source", "task", "column" if expand else "n_columns", "n_tot", "size"]
    if spec["show_class_balance"]:
        cols.append("n_pos")
    cols += [m[0] for m in spec["metrics"]]
    if spec["show_class_balance"]:
        cols.append("ratio")
    else:  # regression: skewness is the label-shape analog of the class-balance ratio
        cols.append("skew")
    # Published-leaderboard reference (best reported model), where known.
    cols += ["leaderboard_score", "leaderboard_metric"]
    cols.append("last_updated")
    return cols


def get_catalog(
    source: str | None = None,
    task: str = "classification",
    expand: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame summarizing the available datasets.

    By default there is **one row per dataset family** (multi-column families collapse to a
    single row with ``n_columns``; metrics are averaged over the columns, while ``n_tot`` and
    ``n_pos`` report the **median** per-column count). Pass ``expand=True`` for **one row per
    label column**.

    The metric columns and class-balance columns depend on ``task``. Every view also
    carries ``leaderboard_score``/``leaderboard_metric`` (the best published result, where
    known — currently MoleculeNet only; blank elsewhere):

    Classification (collapsed) — name, source, task, n_columns, n_tot, n_pos,
        auroc, auprc, ratio, leaderboard_score, leaderboard_metric, last_updated.
        ``expand=True`` swaps ``n_columns``→``column``.
    Regression (collapsed) — name, source, task, n_columns, n_tot, rmse, r2,
        leaderboard_score, leaderboard_metric, last_updated (no n_pos/ratio).
        ``expand=True`` swaps ``n_columns``→``column``.

    ``task="all"`` returns both tasks in one frame with the union of columns (metrics that
    don't apply to a row are NaN).
    """
    if task == "all":
        frames = [
            get_catalog(source=source, task=t, expand=expand)
            for t in ("classification", "regression")
        ]
        cols = catalog_columns("all", expand)
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return pd.DataFrame(columns=cols)
        combined = combined.reindex(columns=cols)
        sort_keys = ["task", "source", "name", "column"] if expand else ["task", "source", "name"]
        return combined.sort_values(sort_keys).reset_index(drop=True)

    spec = _task_metric_spec(task)
    metrics = spec["metrics"]
    balance = spec["show_class_balance"]
    cols = catalog_columns(task, expand)

    rows = []
    sources = list_sources() if source is None else [source]
    for src in sources:
        for info in iter_datasets(src, task):
            meta = info.metadata
            columns = meta.get("columns")
            last_updated = meta.get("last_updated")

            if expand:
                # family columns, or a single synthetic column for legacy datasets.
                items = columns.items() if columns else [(info.dataset, meta)]
                for name, c in items:
                    n_tot, n_pos = c.get("n_samples"), c.get("n_positives")
                    row = {
                        "id": make_id(src, info.dataset, name),
                        "name": info.dataset,
                        "source": src,
                        "task": task,
                        "column": name,
                        "n_tot": n_tot,
                        "size": meta.get("size_bytes"),
                        "leaderboard_score": c.get("leaderboard_value"),
                        "leaderboard_metric": c.get("leaderboard_metric"),
                        "last_updated": last_updated,
                    }
                    for disp, collapsed_key, percol_key in metrics:
                        row[disp] = c.get(percol_key, c.get(collapsed_key))
                    if balance:
                        row["n_pos"] = n_pos
                        row["ratio"] = _ratio(n_tot, n_pos)
                    else:
                        row["skew"] = c.get("skewness")
                    rows.append(row)
                continue

            # collapsed: one row per family. For families, n_tot/n_pos/ratio summarize the
            # per-column values by their median (these are hidden in the collapsed view, so
            # the median is the single most representative value); single-column families
            # trivially reduce to their one column, and legacy datasets use their own counts.
            if columns:
                vals = columns.values()
                n_tot = _median_int(c.get("n_samples") for c in vals)
                n_pos = _median_int(c.get("n_positives") for c in vals)
                ratio = _median_float(
                    _ratio(c.get("n_samples"), c.get("n_positives")) for c in vals
                )
            else:
                n_tot = meta.get("n_samples")
                n_pos = meta.get("n_positives")
                ratio = _ratio(n_tot, n_pos)
            row = {
                "id": make_id(src, info.dataset),
                "name": info.dataset,
                "source": src,
                "task": task,
                "n_columns": len(info.columns),
                "n_tot": n_tot,
                "size": meta.get("size_bytes"),
                "leaderboard_score": meta.get("leaderboard_value"),
                "leaderboard_metric": meta.get("leaderboard_metric"),
                "last_updated": last_updated,
            }
            for disp, collapsed_key, _percol_key in metrics:
                row[disp] = meta.get(collapsed_key)
            if balance:
                row["n_pos"] = n_pos
                row["ratio"] = ratio
            else:
                row["skew"] = (
                    _median_float(c.get("skewness") for c in columns.values())
                    if columns else meta.get("skewness")
                )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=cols)
    sort_keys = ["task", "source", "name", "column"] if expand else ["task", "source", "name"]
    return pd.DataFrame(rows)[cols].sort_values(sort_keys).reset_index(drop=True)


def mirror_dataset(
    source: str,
    dataset: str,
    featurization: str | None = "morgan",
    output_dir: str | os.PathLike = "data",
    task: str = "classification",
    from_dir: str | os.PathLike | None = None,
) -> Path:
    """Mirror a dataset into a local folder.

    Parameters
    ----------
    source : str
        e.g. "tdcommons" or "moleculenet".
    dataset : str
        Dataset name, e.g. "ames".
    featurization : str or None
        One of "morgan", "rdkit", or None.
        Controls which feature matrix is downloaded.
    output_dir : str or PathLike
        Root folder where the dataset should be written. Defaults to ``data``.
    task : str
        "classification" or "regression". Defaults to ``classification``.
    from_dir : str or PathLike or None
        When given, copy the dataset files from this local directory (laid out as
        ``{from_dir}/{source}/{task}/{dataset}/{filename}``, the same layout the prepare
        scripts write under ``data/``) instead of downloading from S3. Useful for testing a
        freshly prepared source before uploading it.

    Returns
    -------
    Path
        Path to the created dataset directory.
    """
    if featurization is not None and featurization not in FEATURIZATIONS:
        raise ValueError(
            f"featurization must be None or one of {list(FEATURIZATIONS)}, got {featurization!r}"
        )

    dataset_dir = _output_dataset_dir(output_dir, source, task, dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def fetch(filename: str) -> None:
        dest = dataset_dir / filename
        if from_dir is not None:
            _copy_from_dir(source, task, dataset, filename, dest, from_dir)
        else:
            _download_to(source, task, dataset, filename, dest)

    try:
        fetch("data.csv")
        fetch("folds.csv")

        if featurization is not None:
            fetch(f"{featurization}.npy")

        # metadata.json (skip-if-exists, like the data files): prefer a copy from --from_dir,
        # else fall back to the packaged metadata.
        metadata_dest = dataset_dir / "metadata.json"
        if not metadata_dest.exists():
            local_metadata = (
                Path(from_dir) / source / task / dataset / "metadata.json"
                if from_dir is not None
                else None
            )
            if local_metadata is not None and local_metadata.exists():
                shutil.copy2(local_metadata, metadata_dest)
            else:
                shutil.copy2(
                    _pkg_data_path(source, task, dataset, "metadata.json"), metadata_dest
                )
    except Exception:
        _rmdir_if_empty_chain(dataset_dir, Path(output_dir))
        raise

    return dataset_dir
