from collections.abc import Iterator
import shutil
import json
import os
from importlib import resources
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

S3_BASE = "https://eosvc-public.s3.amazonaws.com/eosbench/data"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "eosbench")
SUPPORTED_FEATURIZATIONS = (None, "morgan", "chemeleon")

__all__ = [
    "available_datasets",
    "dataset_catalog",
    "Dataset",
    "DatasetInfo",
    "Splits",
    "fetch_datasets",
    "iter_datasets",
    "list_datasets",
    "load_dataset",
]


def _cache_path(source: str, task_type: str, dataset: str, filename: str) -> str:
    path = os.path.join(CACHE_DIR, source, task_type, dataset, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _fetch(source: str, task_type: str, dataset: str, filename: str) -> str:
    dest = _cache_path(source, task_type, dataset, filename)
    _download_to(source, task_type, dataset, filename, dest)
    return dest


def _download_to(
    source: str,
    task_type: str,
    dataset: str,
    filename: str,
    dest: str | os.PathLike,
) -> str:
    dest = str(dest)
    if not os.path.exists(dest):
        url = f"{S3_BASE}/{source}/{task_type}/{dataset}/{filename}"
        try:
            urlretrieve(url, dest)
        except URLError as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e
    return dest


def _pkg_data_path(source: str, task_type: str, dataset: str, filename: str) -> str:
    return str(
        resources.files("eosbench")
        / "_data" / source / task_type / dataset / filename
    )


def _output_dataset_dir(output_dir: str | os.PathLike, source: str, task_type: str, dataset: str) -> Path:
    return Path(output_dir) / source / task_type / dataset


class Splits:
    """Fold-based train/test index splits.

    Supports iteration and direct indexing::

        for train_idxs, test_idxs in dataset.split():
            ...

        train_idxs, test_idxs = dataset.split[0]
    """

    def __init__(self, folds: np.ndarray):
        unique = sorted(set(folds.tolist()))
        self._splits = [
            (np.where(folds != k)[0], np.where(folds == k)[0])
            for k in unique
        ]

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
        SMILES strings (featurization=None) or fingerprint matrix (n, 2048).
    y : np.ndarray
        Target values, shape (n,).
    split : Splits
        Cross-validation splits. Iterable and indexable::

            for train_idxs, test_idxs in dataset.split():
                ...
            train_idxs, test_idxs = dataset.split[0]

    metadata : dict
        Dataset metadata including baseline performance metrics.
    """

    def __init__(self, X, y: np.ndarray, folds: np.ndarray, metadata: dict):
        self.X = X
        self.y = y
        self.split = Splits(folds)
        self.metadata = metadata


class DatasetInfo:
    """Lightweight dataset descriptor — metadata only, no data downloaded.

    Attributes
    ----------
    source : str
    dataset : str
    task_type : str
    metadata : dict
    """

    def __init__(self, source: str, task_type: str, dataset: str):
        self.source = source
        self.task_type = task_type
        self.dataset = dataset
        meta_path = _pkg_data_path(source, task_type, dataset, "metadata.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)

    def __repr__(self):
        return (
            f"DatasetInfo(source={self.source!r}, dataset={self.dataset!r}, "
            f"task_type={self.task_type!r}, n_samples={self.metadata.get('n_samples')})"
        )

    def load(self, featurization: str | None = "morgan") -> Dataset:
        """Download and return the full Dataset."""
        return load_dataset(
            self.source,
            self.dataset,
            featurization,
            task_type=self.task_type,
        )


def load_dataset(
    source: str,
    dataset: str,
    featurization: str | None = "morgan",
    task_type: str = "classification",
) -> Dataset:
    """Load a benchmark dataset.

    Parameters
    ----------
    source : str
        Dataset source: "tdc" or "chembl".
    dataset : str
        Dataset name, e.g. "ames" or "chembl4649948".
    featurization : str or None
        "morgan", "chemeleon", or None.
        If None, X is a list of SMILES strings.
        Otherwise X is a NumPy array.
    task_type : str
        "classification" or "regression" (default: "classification").

    Returns
    -------
    Dataset
    """
    if featurization not in SUPPORTED_FEATURIZATIONS:
        raise ValueError(
            f"featurization must be None, 'morgan', or 'chemeleon', got {featurization!r}"
        )

    csv_path = _fetch(source, task_type, dataset, "data.csv")
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].tolist()
    activity_col = "activity" if "activity" in df.columns else "value"
    y = df[activity_col].to_numpy()

    if featurization is None:
        X = smiles
    else:
        X = np.load(_fetch(source, task_type, dataset, f"{featurization}.npy"))

    folds_path = _pkg_data_path(source, task_type, dataset, "folds.csv")
    folds = pd.read_csv(folds_path)["fold"].values.astype(int)

    meta_path = _pkg_data_path(source, task_type, dataset, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    return Dataset(X, y, folds, metadata)


def iter_datasets(
    source: str,
    task_type: str = "classification",
) -> Iterator[DatasetInfo]:
    """Iterate over available datasets for a source and task type.

    Yields lightweight :class:`DatasetInfo` objects with metadata only —
    no data is downloaded.

    Parameters
    ----------
    source : str
        "tdc" or "chembl".
    task_type : str
        "classification" or "regression" (default: "classification").

    Yields
    ------
    DatasetInfo
    """
    base = resources.files("eosbench") / "_data" / source / task_type
    try:
        entries = sorted(entry.name for entry in base.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    for name in entries:
        yield DatasetInfo(source, task_type, name)


def list_datasets(source: str | None = None, task_type: str = "classification") -> list[dict]:
    """List available datasets.

    Parameters
    ----------
    source : str or None
        Filter by "tdc" or "chembl". If None, returns all.
    task_type : str
        "classification" or "regression" (default: "classification").

    Returns
    -------
    list of dicts with keys: source, dataset, task_type.
    """
    sources = ["tdc", "chembl"] if source is None else [source]
    result = []
    for src in sources:
        for info in iter_datasets(src, task_type):
            result.append(
                {
                    "source": src,
                    "dataset": info.dataset,
                    "task_type": task_type,
                }
            )
    return sorted(result, key=lambda d: (d["source"], d["dataset"]))


def available_datasets(
    source: str | None = None,
    task_type: str = "classification",
) -> list[str]:
    """Return the full sorted list of available dataset names."""
    datasets = [entry["dataset"] for entry in list_datasets(source=source, task_type=task_type)]
    return sorted(datasets)


def dataset_catalog(
    source: str | None = None,
    task_type: str = "classification",
) -> pd.DataFrame:
    """Return a DataFrame summarizing the available datasets.

    Columns
    -------
    name
        Dataset name.
    source
        Dataset source.
    samples
        Number of samples.
    auroc
        Mean AUROC, if available.
    aupr
        Mean AUPR, if available.
    n_pos
        Number of positive samples, if available.
    n_tot
        Total number of samples.
    ratio
        Positive class ratio, if available.
    """
    rows = []
    sources = ["tdc", "chembl"] if source is None else [source]
    for src in sources:
        for info in iter_datasets(src, task_type):
            metadata = info.metadata
            n_tot = metadata.get("n_samples")
            n_pos = metadata.get("n_positives")
            ratio = None
            if n_tot not in (None, 0) and n_pos is not None:
                ratio = n_pos / n_tot

            rows.append(
                {
                    "name": info.dataset,
                    "source": info.source,
                    "samples": n_tot,
                    "auroc": metadata.get("auroc_mean"),
                    "aupr": metadata.get("aupr_mean"),
                    "n_pos": n_pos,
                    "n_tot": n_tot,
                    "ratio": ratio,
                }
            )

    return pd.DataFrame(rows).sort_values(["source", "name"]).reset_index(drop=True)


def fetch_datasets(
    task_names: list[str],
    output_dir: str | os.PathLike = "data",
    task_type: str = "classification",
) -> list[Path]:
    """Fetch datasets into a local folder tree.

    Parameters
    ----------
    task_names : list[str]
        Dataset names to fetch, for example ``["ames", "herg"]``.
    output_dir : str or PathLike
        Root folder where datasets should be materialized. Defaults to ``data``.
    task_type : str
        Dataset task type, defaults to ``classification``.

    Returns
    -------
    list[Path]
        Paths to the created dataset directories.
    """
    catalog = dataset_catalog(task_type=task_type)
    created_dirs: list[Path] = []

    for task_name in task_names:
        matches = catalog.loc[catalog["name"] == task_name, "source"].tolist()
        if not matches:
            raise ValueError(f"Unknown dataset: {task_name!r}")
        if len(matches) > 1:
            raise ValueError(
                f"Dataset name {task_name!r} is ambiguous across sources: {matches}"
            )

        source = matches[0]
        dataset_dir = _output_dataset_dir(output_dir, source, task_type, task_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("data.csv", "morgan.npy", "chemeleon.npy"):
            _download_to(source, task_type, task_name, filename, dataset_dir / filename)

        for filename in ("metadata.json", "folds.csv"):
            src_path = _pkg_data_path(source, task_type, task_name, filename)
            shutil.copy2(src_path, dataset_dir / filename)

        created_dirs.append(dataset_dir)

    return created_dirs
