# Ersilia benchmarks for ML training

A Python package for loading molecular activity datasets used in Ersilia ML development.

> **Important:** This repository is not an official benchmark. Cross-validation splits are arbitrary and were not designed to reproduce any published protocol. Performance numbers obtained here **cannot be directly compared** with results from TDC or any other benchmarking framework.

---

## Installation

Requires Python 3.10+.

```bash
pip install .
```

---

## How to use

The typical use case is benchmarking a machine learning model across multiple datasets. The recommended workflow is:

1. **Browse available datasets** to decide which ones you need:
   ```bash
   eosbench catalog
   ```

2. **Navigate to your working directory** and fetch the datasets you want locally:
   ```bash
   cd my_project/
   eosbench fetch --source tdc --dataset ames --featurization morgan
   eosbench fetch --source tdc --dataset herg --featurization morgan
   ```
   This downloads `data.csv`, `folds.csv`, `morgan.npy`, and `metadata.json` into `./tdc/classification/ames/` (and `herg/`). Files are never re-downloaded if already present.

3. **Load and evaluate** in Python:
   ```python
   from eosbench import load_dataset
   from sklearn.metrics import roc_auc_score

   dataset = load_dataset("tdc", "ames", featurization="morgan")

   aurocs = []
   for train_idx, test_idx in dataset.split():
       model.fit(dataset.X[train_idx], dataset.y[train_idx])
       y_hat = model.predict_proba(dataset.X[test_idx])[:, 1]
       aurocs.append(roc_auc_score(dataset.y[test_idx], y_hat))

   print(f"AUROC: {sum(aurocs)/len(aurocs):.4f}")
   ```

The `load_dataset` function uses the same local cache (`~/.cache/eosbench/`) as `eosbench fetch`, so files you have already downloaded are reused automatically.

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
| `featurization` | `"morgan"`, `"chemeleon"`, `"rdkit"`, `"cddd"`, `None` | feature representation; `None` returns raw SMILES |
| `task_type` | `"classification"`, `"regression"` | defaults to `"classification"` |

**Returns** a `Dataset` object with:

- `dataset.X` — NumPy array or list of SMILES strings if `featurization=None`; shapes: `(n, 2048)` for `morgan`/`chemeleon`, `(n, 217)` for `rdkit`, `(n, 512)` for `cddd`
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
        rdkit.npy       # only if featurization="rdkit"
        cddd.npy        # only if featurization="cddd"
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

## CLI

`eosbench` includes a command-line interface installed alongside the package.

### `eosbench catalog`

Print a table of all available datasets with metadata.

```bash
eosbench catalog                    # all sources
eosbench catalog --source tdc       # filter by source
eosbench catalog --source chembl --task classification
```

### `eosbench info`

Show metadata for a single dataset.

```bash
eosbench info --source tdc --dataset ames
```

Output:

```
        tdc/ames
┌─────────────┬─────────────────┐
│ source      │ tdc             │
│ dataset     │ ames            │
│ task        │ classification  │
│ n_samples   │ 7278            │
│ n_positives │ 3974            │
│ n_negatives │ 3304            │
│ auroc       │ 0.9029 ± 0.0079 │
│ aupr        │ 0.9132 ± 0.0081 │
└─────────────┴─────────────────┘
```

### `eosbench fetch`

Download a dataset to a local folder.

```bash
eosbench fetch --source tdc --dataset ames
eosbench fetch --source tdc --dataset ames --featurization rdkit --output-dir my_data
eosbench fetch --source chembl --dataset chembl4649948 --featurization none
```

| argument | default | description |
|----------|---------|-------------|
| `--source` | required | `tdc` or `chembl` |
| `--dataset` | required | dataset name, e.g. `ames` |
| `--featurization` | `morgan` | `morgan`, `chemeleon`, `rdkit`, `cddd`, or `none` |
| `--output-dir` | `.` | root folder to write into |
| `--task` | `classification` | `classification` or `regression` |

---

## About Ersilia

The [Ersilia Open Source Initiative](https://ersilia.io) builds open tools and models for AI-enabled drug discovery, with a focus on accessibility and global health impact.

![Ersilia Logo](assets/Ersilia_Brand.png)
