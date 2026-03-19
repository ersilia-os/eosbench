# Benchmarks datasets for Ersilia ML development

`eosbench` is a Python package for loading benchmark datasets used in the Ersilia ecosystem. It provides a small API for discovering available datasets, reading packaged metadata and fold assignments, and downloading feature matrices on demand.

The repository currently includes packaged metadata and cross-validation folds for a set of `tdc` and `chembl` classification benchmarks. Dataset payloads are fetched lazily from a public S3 bucket and cached locally the first time they are requested.

## Installation

Use Python 3.10 or newer.

```bash
pip install .
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Start

```python
from eosbench import (
    available_datasets,
    dataset_catalog,
    fetch_datasets,
    list_datasets,
    load_dataset,
)

print(available_datasets())
print(dataset_catalog())
print(list_datasets(source="tdc")[:3])
print(fetch_datasets(["ames"]))

dataset = load_dataset(
    source="tdc",
    dataset="ames",
    featurization="morgan",
    task_type="classification",
)

print(dataset.X.shape)
print(dataset.y.shape)
print(dataset.metadata["auroc_mean"])

train_idx, test_idx = dataset.split[0]
```

To load raw SMILES strings instead of precomputed features:

```python
dataset = load_dataset("tdc", "ames", featurization=None)
print(dataset.X[:5])
```

## Public API

The package root exposes:

- `available_datasets(source=None, task_type="classification")`
- `dataset_catalog(source=None, task_type="classification")`
- `fetch_datasets(task_names, output_dir="data", task_type="classification")`
- `list_datasets(source=None, task_type="classification")`
- `load_dataset(source, dataset, featurization="morgan", task_type="classification")`

Use `available_datasets()` when you just want the full list of dataset names:

```python
from eosbench import available_datasets

print(available_datasets())
print(available_datasets(source="tdc"))
```

Use `dataset_catalog()` when you want a pandas DataFrame with dataset-level summary statistics:

```python
from eosbench import dataset_catalog

catalog = dataset_catalog()
print(catalog.head())
```

The DataFrame columns are:

- `name`
- `source`
- `samples`
- `auroc`
- `aupr`
- `n_pos`
- `n_tot`
- `ratio`

Use `fetch_datasets()` when you want the on-disk folder layout materialized locally:

```python
from eosbench import fetch_datasets

fetch_datasets(["ames", "herg"])
fetch_datasets(["ames"], output_dir="data")
```

This creates a directory tree like:

```text
data/
  tdc/
    classification/
      ames/
        data.csv
        morgan.npy
        chemeleon.npy
        metadata.json
        folds.csv
```

`load_dataset(...)` returns a `Dataset` object with:

- `X`: SMILES strings or a NumPy feature matrix
- `y`: labels
- `split`: iterable and indexable train/test fold splits
- `metadata`: dataset metadata

Supported `featurization` values are:

- `None`
- `"morgan"`
- `"chemeleon"`

## Lower-Level Module API

The `eosbench.dataset` module also exposes:

- `iter_datasets(...)`
- `DatasetInfo`
- `Dataset`
- `Splits`

Example:

```python
from eosbench.dataset import iter_datasets

for info in iter_datasets("tdc"):
    print(info.dataset, info.metadata["n_samples"])
```

## Data Layout

Stored in the repository:

- `src/eosbench/_data/**/metadata.json`
- `src/eosbench/_data/**/folds.csv`

Downloaded on demand and cached under `~/.cache/eosbench/`:

- `data.csv`
- `morgan.npy`
- `chemeleon.npy`

## Testing

```bash
PYTHONPATH=src pytest
```

## About Ersilia

The [Ersilia Open Source Initiative](https://ersilia.io) builds open tools and models for AI-enabled drug discovery, with a strong focus on accessibility and reproducibility.

![Ersilia Logo](assets/Ersilia_Brand.png)
