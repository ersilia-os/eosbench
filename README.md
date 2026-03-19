# Ersilia benchmarks for ML training

A Python package for loading molecular activity datasets used in Ersilia ML development.

> **Important:** This repository is not an official benchmark. Cross-validation splits are arbitrary and were not designed to reproduce any published protocol. Performance numbers obtained here **cannot be directly compared** with results from TDC or any other benchmarking framework.

---

## Installation

Requires Python 3.10+.

```bash
pip install .
```

```bash
pip install -e .[dev]  # for development
```

---

## Datasets

`eosbench` ships with metadata for 20 classification datasets from two sources:

- **TDC** — 18 datasets (ADMET tasks such as Ames, hERG, BBB, CYP450, etc.)
- **ChEMBL** — 2 datasets (large-scale bioactivity datasets)

Dataset files (SMILES, labels, feature matrices, fold assignments) are downloaded on demand from a public S3 bucket and cached under `~/.cache/eosbench/`.

---

## API

There are four functions in the public API.

---

### `get_catalog`

Returns a summary of all available datasets as a pandas DataFrame.

```python
from eosbench import get_catalog

catalog = get_catalog()
catalog = get_catalog(source="tdc")           # filter by source
catalog = get_catalog(task="classification")  # filter by task type
```

Columns:

| column | description |
|--------|-------------|
| `name` | dataset name |
| `source` | `"tdc"` or `"chembl"` |
| `task` | `"classification"` or `"regression"` |
| `n_tot` | total number of samples |
| `n_pos` | number of positive samples |
| `auroc` | mean AUROC across folds |
| `aupr` | mean AUPR across folds |
| `ratio` | positive class ratio |

---

### `load_dataset`

Downloads (if needed) and returns a dataset ready for model training.

```python
from eosbench import load_dataset

dataset = load_dataset("tdc", "ames", featurization="morgan")
```

**Arguments:**

| argument | values | description |
|----------|--------|-------------|
| `source` | `"tdc"`, `"chembl"` | dataset source |
| `dataset` | e.g. `"ames"` | dataset name |
| `featurization` | `"morgan"`, `"chemeleon"`, `None` | feature representation; `None` returns raw SMILES |
| `task_type` | `"classification"`, `"regression"` | defaults to `"classification"` |

**Returns** a `Dataset` object with:

- `dataset.X` — NumPy array `(n, 2048)` or list of SMILES strings if `featurization=None`
- `dataset.y` — NumPy array of labels, shape `(n,)`
- `dataset.split` — cross-validation splits (iterable and indexable)
- `dataset.metadata` — dict with dataset statistics

```python
# iterate over folds
for train_idx, test_idx in dataset.split:
    X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]

# or index directly
train_idx, test_idx = dataset.split[0]
```

---

### `mirror_dataset`

Downloads a dataset to a local folder on disk. Useful when you want the raw files rather than a Python object. Mirrors the signature of `load_dataset`.

```python
from eosbench import mirror_dataset

mirror_dataset("tdc", "ames", featurization="morgan")
mirror_dataset("tdc", "ames", featurization="chemeleon")  # adds chemeleon.npy, skips existing files
mirror_dataset("tdc", "ames", featurization=None, output_dir="my_data")  # no feature matrix
```

Files already present in the destination are never re-downloaded or overwritten.

**Arguments** are the same as `load_dataset`, plus:

| argument | description |
|----------|-------------|
| `output_dir` | root folder to write into (default: `"data"`) |

**Returns** the `Path` to the dataset directory.

The resulting folder layout:

```
data/
  tdc/
    classification/
      ames/
        data.csv
        folds.csv
        morgan.npy      # only if featurization="morgan"
        chemeleon.npy   # only if featurization="chemeleon"
        metadata.json
```

---

### `get_path`

Returns the path where a dataset lives (or would live) within a base directory. Useful for building file paths without hardcoding the folder structure.

```python
from eosbench import get_path

path = get_path("data", "ames", source="tdc", task="classification")
# Path("data/tdc/classification/ames")
```

---

## Testing

```bash
pytest
```

---

## About Ersilia

The [Ersilia Open Source Initiative](https://ersilia.io) builds open tools and models for AI-enabled drug discovery, with a focus on accessibility and global health impact.

![Ersilia Logo](assets/Ersilia_Brand.png)
