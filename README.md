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
   eosbench fetch --source tdcommons --dataset ames --featurization morgan
   eosbench fetch --source tdcommons --dataset herg --featurization morgan
   ```
   This downloads `data.csv`, `folds.csv`, `morgan.npy`, and `metadata.json` into `./tdcommons/classification/ames/` (and `herg/`). Files are never re-downloaded if already present.

3. **Load and evaluate** in Python:
   ```python
   from eosbench import load_dataset
   from sklearn.metrics import roc_auc_score

   dataset = load_dataset("tdcommons", "ames", featurization="morgan")

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

Browse everything online at **[ersilia-os.github.io/eosbench](https://ersilia-os.github.io/eosbench/)** (see [Catalog website](#catalog-website)).

`eosbench` ships with metadata for classification datasets from several sources:

- **tdcommons** — single-input SMILES binary-classification datasets from [Therapeutics Data Commons](https://tdcommons.ai/): ADMET properties (Ames, hERG, BBB, CYP450, …) plus HTS bioassays (SARS-CoV-2, Butkiewicz panel, HIV), spanning ~880 to ~340k molecules. Built by `scripts/prepare_tdcommons.py`
- **MoleculeNet** — BBBP, BACE, HIV, Tox21, ClinTox, SIDER, MUV, ToxCast
- **Polaris** — single-input, binary-classification benchmarks auto-discovered from the [Polaris Hub](https://polarishub.io/)

Datasets are organized as **families**. A family is a collection of one or more binary
label **columns** (endpoints) over a *shared* set of molecules. Single-column sets
(BBBP, BACE, HIV) are 1-column families; multi-column sets (Tox21, ClinTox, …) keep every
endpoint as a column within one family. All columns of a family share the same molecules,
feature matrices, and a single **conserved** train/test split — so a molecule lands on the
same side for every column — with NaN labels where a given column is unmeasured.

The vocabulary, end to end: **source** (`tdcommons`, `moleculenet`, …) → **dataset** (the family,
e.g. `tox21`) → **column** (an endpoint, e.g. `NR-AR`); **task** is the ML problem type
(`classification` / `regression`).

Dataset files (SMILES, labels, feature matrices, fold assignments) are downloaded on demand from a public S3 bucket and cached under `~/.cache/eosbench/`.

---

## API

The main public functions are described below.

---

### `get_catalog`

Returns a summary of all available datasets as a pandas DataFrame.

By default there is **one row per family** (multi-column families collapse to a single row).
Pass `expand=True` for **one row per column**.

```python
from eosbench import get_catalog, list_columns

catalog = get_catalog()                       # one row per family
catalog = get_catalog(source="tdcommons")     # filter by source
catalog = get_catalog(expand=True)            # one row per label column

list_columns("moleculenet", "tox21")          # ["NR-AR", "NR-AR-LBD", ...]
```

The metric columns depend on `task` — classification reports `auroc`/`auprc`,
regression reports `rmse`/`r2`, and the class-balance columns (`n_pos`, `ratio`)
appear for classification only.

Columns (collapsed, classification):

| column | description |
|--------|-------------|
| `name` | family name (shown as `dataset` in the CLI table) |
| `source` | `"tdcommons"`, `"moleculenet"`, `"polaris"` |
| `task` | `"classification"` or `"regression"` |
| `n_columns` | number of label columns in the family |
| `n_tot` | total molecules (or samples for single-column) |
| `n_pos` | positive samples (classification only) |
| `auroc` | mean baseline AUROC (averaged over columns) |
| `auprc` | mean baseline AUPRC (averaged over columns) |
| `ratio` | positive class ratio, `n_pos / n_tot` (classification only) |
| `leaderboard_score` | best published result, where known (`lb_score` in the CLI table) |
| `leaderboard_metric` | metric `leaderboard_score` is measured in, e.g. `ROC-AUC` (`lb_metric` in the CLI) |
| `last_updated` | date the family was last prepared (ISO `YYYY-MM-DD`) |

For `task="regression"` the `auroc`/`auprc` columns are replaced by `rmse`/`r2`
and `n_pos`/`ratio` are omitted. With `expand=True` the frame has one row per
column: `name` (family), `column` (endpoint), `n_tot`, the task metrics, the
leaderboard columns, and `last_updated` (plus `n_pos`/`ratio` for classification).

> **What the metrics mean:** `auroc`/`auprc` (and `rmse`/`r2`) are a **RandomForest
> baseline averaged over random K-fold cross-validation** — a reference floor for
> "how hard is this dataset," not the best published model. The
> `leaderboard_score`/`leaderboard_metric` columns *are* the best published model
> (MoleculeNet, and `tdcommons` ADMET tasks sourced from Polaris; blank where unknown).

---

### `load_dataset`

Downloads (if needed) and returns a dataset ready for model training.

```python
from eosbench import load_dataset

dataset = load_dataset("tdcommons", "ames", featurization="morgan")
```

**Arguments:**

| argument | values | description |
|----------|--------|-------------|
| `source` | `"tdcommons"`, `"moleculenet"`, `"polaris"` | dataset source |
| `dataset` | e.g. `"ames"`, `"tox21"` | family name |
| `featurization` | `"morgan"`, `"rdkit"`, `None` | feature representation; `None` returns raw SMILES |
| `task` | `"classification"`, `"regression"` | ML task type; defaults to `"classification"` |
| `split` | `"random"`, `"scaffold"` | `"random"` (default) gives K-fold CV; `"scaffold"` gives a single predefined train/test holdout (where available) |
| `column` | e.g. `"NR-AR"` | which label column of a multi-column family to load; `None` (default) picks the sole column of a single-column family and raises (listing columns) for a multi-column one |

```python
# a multi-column family: pick the endpoint, rows unlabeled for that column are dropped
dataset = load_dataset("moleculenet", "tox21", column="NR-AR", split="scaffold")
```

```python
# random K-fold cross-validation (default)
dataset = load_dataset("moleculenet", "bbbp", featurization="morgan", split="random")
for train_idx, test_idx in dataset.split():   # one pair per fold
    ...

# predefined scaffold holdout
dataset = load_dataset("moleculenet", "bbbp", featurization="morgan", split="scaffold")
train_idx, test_idx = dataset.split[0]         # a single train/test pair
```

**Returns** a `Dataset` object with:

- `dataset.X` — NumPy array or list of SMILES strings if `featurization=None`; shapes: `(n, 2048)` for `morgan`, `(n, 217)` for `rdkit`
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

mirror_dataset("tdcommons", "ames", featurization="morgan")
mirror_dataset("tdcommons", "ames", featurization="rdkit")  # adds rdkit.npy, skips existing files
mirror_dataset("tdcommons", "ames", featurization=None, output_dir="my_data")  # no feature matrix
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
  tdcommons/
    classification/
      ames/
        data.csv
        folds.csv
        morgan.npy      # only if featurization="morgan"
        rdkit.npy       # only if featurization="rdkit"
        metadata.json
```

---

### `get_path`

Returns the path where a dataset lives (or would live) within a base directory. Useful for building file paths without hardcoding the folder structure.

```python
from eosbench import get_path

path = get_path("data", "ames", source="tdcommons", task="classification")
# Path("data/tdcommons/classification/ames")
```
---

## CLI

`eosbench` includes a command-line interface installed alongside the package.

### `eosbench catalog`

Print a table of all available datasets with metadata. The table adapts to the
task: classification shows `auroc`/`auprc` (plus `n_pos`/`ratio`), regression
shows `rmse`/`r2`. Counts are shown as grouped integers (e.g. `12,665`).

```bash
eosbench catalog                                  # all sources
eosbench catalog --source tdcommons                     # filter by source
eosbench catalog --task regression --sort_by rmse # regression sets, lowest RMSE first
```

Besides `--source`/`--task`/`--expand`, the catalog can be filtered, sorted and
limited:

- **filters** (combine with AND): `--name`, `--min/--max_samples`,
  `--min/--max_ratio`, `--min/--max_auroc`, `--min/--max_auprc` (classification),
  `--min/--max_rmse`, `--min/--max_r2` (regression). Threshold filters skip
  datasets with a missing value for that column.
- **sorting/limiting**: `--sort_by COLUMN [--desc]` and `--limit N`.

```bash
eosbench catalog --min_samples 1000 --min_auroc 0.8 --sort_by auroc --desc
eosbench catalog --name cyp --limit 5
```

See `eosbench catalog --help` for the full flag and column reference. Metric
values are a RandomForest baseline averaged over random K-fold CV — a reference
floor, not the best published model.

### `eosbench info`

Show metadata for a single dataset.

```bash
eosbench info --source tdcommons --dataset ames
```

Output:

```
                         tdcommons/ames
┌────────────────────┬──────────────────────────────────────────┐
│ source             │ tdcommons                                │
│ dataset            │ ames                                     │
│ task               │ classification                           │
│ n_molecules        │ 7278                                     │
│ n_columns          │ 1                                        │
│ leaderboard        │ 0.8710 (AUROC)                           │
│ leaderboard_split  │ scaffold                                 │
│ leaderboard_source │ ZairaChem, Polaris TDC ADMET leaderboard │
│ last_updated       │ 2026-06-16                               │
└────────────────────┴──────────────────────────────────────────┘
                                  columns
┏━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━ … ━┓
┃ column ┃ n    ┃ pos  ┃ neg  ┃ auroc (random)  ┃ auprc (random)  ┃     ┃
┡━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━ … ━┩
│ ames   │ 7278 │ 3974 │ 3304 │ 0.9091 ± 0.0070 │ 0.9199 ± 0.0066 │ …   │
└────────┴──────┴──────┴──────┴─────────────────┴─────────────────┴─ … ─┘
auroc/auprc are a RandomForest baseline (reference floor, not the best
published model); the leaderboard line is the best published model (Polaris).
```

Every dataset is a **family**, so `info` shows a summary plus a per-column table
(`tdcommons` and `moleculenet` ADMET tasks are 1-column families).

For a multi-column family, `eosbench info` prints a summary plus a per-column
table (with descriptions truncated to fit). To see **one column's full,
untruncated details** — every metric, the complete description, and any other
metadata fields — pass `--column`:

```bash
eosbench info --source moleculenet --dataset tox21 --column NR-AhR
```

### `eosbench fetch`

Download a dataset to a local folder.

```bash
eosbench fetch --source tdcommons --dataset ames
eosbench fetch --source tdcommons --dataset ames --featurization rdkit --output_dir my_data
eosbench fetch --source moleculenet --dataset bbbp --featurization none
# copy from a locally prepared mirror instead of S3 (e.g. before uploading)
eosbench fetch --source polaris --dataset bbb-martins --from_dir data
```

| argument | default | description |
|----------|---------|-------------|
| `--source` | required | `tdcommons`, `moleculenet`, or `polaris` |
| `--dataset` | required | dataset name, e.g. `ames` |
| `--featurization` | `morgan` | `morgan`, `rdkit`, or `none` |
| `--output_dir` | `.` | root folder to write into |
| `--task` | `classification` | `classification` or `regression` |
| `--from_dir` | `None` | copy from a local mirror laid out as `DIR/{source}/{task}/{dataset}/` instead of downloading from S3 |

---

## Preparing new datasets

New datasets are built with the scripts under `scripts/`. These require extra tooling
(rdkit, scikit-learn) that is **not** needed to consume eosbench:

```bash
pip install -e ".[prepare]"
```

To add MoleculeNet datasets:

```bash
python scripts/prepare_moleculenet.py                      # default subset
python scripts/prepare_moleculenet.py --datasets bbbp,tox21
python scripts/prepare_moleculenet.py --datasets all --n_folds 5 --seed 42
```

For each set this writes **one family**:

- downloads the raw CSV from the public DeepChem mirror,
- keeps every binary endpoint as a label column (NaN where unlabeled); single-column sets
  become 1-column families,
- computes `morgan` and `rdkit` features **once** for the shared molecules,
- writes a single **conserved** split into `folds.csv` — a random K-fold (`random_fold`)
  and a deterministic Bemis–Murcko scaffold holdout (`scaffold_split`) — shared by all
  columns (whole scaffold groups stay together, so train and test never share a scaffold).
  For single-column families the split is **class-stratified** so every fold and the scaffold
  test set contain both classes (a naive scaffold split puts the rare singleton scaffolds in
  the test tail, which on skewed sets like BBBP gives a single-class test set with undefined
  AUROC). Multi-column families have no single label to stratify on, so they use a plain
  conserved split and the baseline simply skips any column whose restricted test set is
  single-class,
- trains a per-column RandomForest baseline and records **both AUROC and AUPR** (random CV
  mean ± std and the scaffold-holdout score) per column, plus the published leaderboard
  reference and a `last_updated` date in `metadata.json`. Use `--no_baseline` to skip the
  baseline (much faster for large families like ToxCast/MUV).

To add Polaris datasets (requires a Polaris Hub login):

```bash
polaris login                                              # one-time, cached token
python scripts/prepare_polaris.py                          # auto-discover all qualifying benchmarks
python scripts/prepare_polaris.py --datasets polaris/some-benchmark
python scripts/prepare_polaris.py --limit 5 --no_baseline  # quick pass
```

This enumerates Polaris Hub benchmarks and keeps the **single-input, binary-classification**
ones (multi-input, regression, and multiclass benchmarks are skipped, each logged with a
reason). Single-target benchmarks become 1-column families; single-input multi-target
benchmarks become multi-column families. Polaris hides test labels only behind its split API —
the labels live in the underlying dataset table, which the script reads directly — and unlike
MoleculeNet the `scaffold_split` column carries Polaris's **official** train/test split
(`scaffold_split_method: "polaris"`) rather than a computed scaffold split; a random K-fold is
added alongside.

To add TDC (Therapeutics Data Commons) datasets:

```bash
python scripts/prepare_tdcommons.py                       # all binary ADMET datasets
python scripts/prepare_tdcommons.py --datasets ames,herg  # specific datasets
python scripts/prepare_tdcommons.py --limit 5 --no_baseline
```

This auto-discovers the single-prediction **ADME**, **Tox** and **HTS** (high-throughput
screening bioassay) datasets via `PyTDC`, keeps the **binary-classification** ones (regression
datasets are skipped, each logged), and writes one 1-column family per dataset under the
**`tdcommons`** source. Sizes range from ~880 molecules to ~340k, giving benchmarks of varied
scale. Other `single_pred` groups are excluded on purpose (QM/Yields are regression;
Epitope/Paratope/Develop/CRISPROutcome take protein/sequence inputs, not SMILES). Like Polaris, it honours
**TDC's own scaffold split** as the holdout (`scaffold_split_method: "tdc-scaffold"`) and adds
a random K-fold CV. Crucially, leaderboard references are sourced from **Polaris**, not TDC:
the script reads `scripts/tdcommons_leaderboard.json` (curated from the Polaris-hosted TDC
ADMET leaderboard, whose values are more reliable than TDC's own) and attaches them where
available; other datasets stay blank. (`tdcommons` replaces the older `tdc` source.)

Files are written to `data/{source}/classification/{family}/` (one multi-column `data.csv`,
one `folds.csv`, one feature matrix per featurizer) and a copy of `metadata.json` is bundled
under `src/eosbench/_data/...` so the family shows up in the catalog. The heavy files are
**not** uploaded automatically — the script prints the `aws s3` command to publish them.

The shared core lives in `scripts/_prepare_common.py` so additional sources can be added
as `scripts/prepare_<source>.py` that reuse `prepare_family(...)`. Sources that ship their own
train/test split (like Polaris) pass it via `prepare_family(..., holdout=..., holdout_method=...)`.

### Leaderboard references

The `leaderboard_score`/`leaderboard_metric` columns record the best **published** result
per dataset (distinct from the RandomForest baseline). These come from curated snapshots:

- **MoleculeNet** — `scripts/moleculenet_leaderboard.json`, applied at prep time by
  `prepare_moleculenet.py`.
- **tdcommons** — `scripts/tdcommons_leaderboard.json`, applied at prep time by
  `prepare_tdcommons.py`. Values are sourced from the **Polaris Hub** (which hosts the TDC
  ADMET Benchmark Group), *not* from TDC's own leaderboard, which can contain errors. Each
  task keeps its official metric (AUROC, or AUPRC for the imbalanced CYP tasks). Datasets
  outside the official ADMET Benchmark Group (e.g. `clintox`, `cyp1a2_veith`) are left blank.

---

## Catalog website

**Browse the catalog online: [ersilia-os.github.io/eosbench](https://ersilia-os.github.io/eosbench/)**

The catalog is also browsable as a static website, published with **GitHub Pages**:
a single page with client-side search, sorting, and filters over every dataset.

`scripts/export_catalog.py` reads the bundled metadata (the same source as `eosbench catalog`)
across all sources and both tasks, writes `site/catalog.json`, and **HEAD-probes each dataset's
`data.csv` on the public S3 bucket** so each row shows an *available* / *pending* badge — i.e. it
reflects what is actually fetchable, not just what has metadata.

The `.github/workflows/pages.yml` workflow rebuilds and redeploys the site on every push to
`main` and on a daily schedule (the cron keeps the availability badges current even when S3 is
updated without a commit). To preview locally:

```bash
python scripts/export_catalog.py        # writes site/catalog.json (with live S3 checks)
python -m http.server -d site           # open http://localhost:8000
```

> **One-time setup:** in the GitHub repo, **Settings → Pages → Build and deployment → Source =
> GitHub Actions**. After that the workflow publishes automatically.

---

## About Ersilia

The [Ersilia Open Source Initiative](https://ersilia.io) builds open tools and models for AI-enabled drug discovery, with a focus on accessibility and global health impact.

![Ersilia Logo](assets/Ersilia_Brand.png)
