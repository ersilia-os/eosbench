# Ersilia benchmarks for ML training

> **Work in progress:** this repository is under active development. Datasets, APIs, and
> documentation may change without notice.

A Python package for loading molecular activity datasets used in Ersilia ML development.

> **Important:** This repository is not an official benchmark. The `baseline` metric (random K-fold CV) uses an arbitrary split not designed to reproduce any published protocol, and is **not comparable** to outside results. The scaffold-holdout metric (`scaffold_auroc`/`scaffold_aupr`/`scaffold_rmse`/`scaffold_r2`) is a different story for `tdcommons`/`polaris` datasets specifically — it honours those sources' own official split — but whether a given published `leaderboard` number is a fair comparison to it still varies row by row; see `eosbench catalog`'s `leaderboard` glyph or [Leaderboard references](#leaderboard-references) before treating any number here as equivalent to a paper's.

---

## Installation

Requires Python 3.10+.

```bash
conda create -n eosbench python=3.12
conda activate eosbench

git clone git@github.com:ersilia-os/eosbench.git
cd eosbench
pip install -e .
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

`eosbench` ships with metadata for classification and regression datasets from several sources:

- **tdcommons** — single-input SMILES binary-classification datasets from [Therapeutics Data Commons](https://tdcommons.ai/): ADMET properties (Ames, hERG, BBB, CYP450, …) plus HTS bioassays (SARS-CoV-2, Butkiewicz panel, HIV), spanning ~880 to ~340k molecules. Built by `scripts/prepare_tdcommons.py`
- **MoleculeNet** — *classification*: BBBP, BACE, HIV, Tox21, ClinTox, SIDER, MUV, ToxCast. *Regression*: ESOL, FreeSolv, Lipophilicity (single-target solubility/logD/hydration) and QM8/QM9 (multi-target quantum properties). Regression sets record a RandomForest **RMSE/R²** baseline, with curated RMSE leaderboard references for the three solubility-type sets.
- **polaris** — the 13 single-input binary-classification benchmarks from the **Polaris Hub**
  (`polarishub.io`) under owner `tdcommons` — the same 13 ADMET datasets as above, but
  prepared directly from Polaris rather than PyTDC, honouring Polaris's own official split
  (verified byte-for-byte identical to `tdcommons`'s own scaffold holdout for the same
  dataset). A deliberate, permanent second copy: it exists to carry Polaris Hub's own, genuinely
  live leaderboard (see [Leaderboard references](#leaderboard-references)) — a different,
  separately-populated leaderboard from TDC's own, not a mirror of it. Built by
  `scripts/prepare_polaris.py`; Polaris also hosts 9 further ADMET *regression* benchmarks
  under the same owner, not yet prepared here since eosbench's Polaris support only handles
  classification so far.

Datasets are organized as **families**. A family is a collection of one or more binary
label **columns** (endpoints) over a *shared* set of molecules. Single-column sets
(BBBP, BACE, HIV) are 1-column families; multi-column sets (Tox21, ClinTox, …) keep every
endpoint as a column within one family. All columns of a family share the same molecules,
feature matrices, and a single **conserved** train/test split — so a molecule lands on the
same side for every column — with NaN labels where a given column is unmeasured.

The vocabulary, end to end: **source** (`tdcommons`, `moleculenet`, `polaris`) → **dataset**
(the family, e.g. `tox21`) → **column** (an endpoint, e.g. `NR-AR`); **task** is the ML
problem type (`classification` / `regression`).

Dataset files (SMILES, labels, feature matrices, fold assignments) are downloaded on demand from a public S3 bucket and cached under `~/.cache/eosbench/`.

---

## API

`eosbench` also exposes a small Python API (`get_catalog`, `load_dataset`, `list_columns`)
mirroring the CLI below, used in the [How to use](#how-to-use) example above. It isn't fully
documented in this README yet — for now, the CLI is the more complete reference, so prefer it
for browsing and fetching datasets.

---

## CLI

`eosbench` includes a command-line interface installed alongside the package.

### `eosbench catalog`

Print a table of all available datasets with metadata. The table adapts to the
task, and counts are shown as grouped integers (e.g. `12,665`).

```bash
eosbench catalog                                  # all sources
eosbench catalog --source tdcommons                     # filter by source
eosbench catalog --task regression --sort_by rmse # regression sets, lowest RMSE first
```

Columns:

| column | meaning |
|--------|---------|
| `id` | deterministic 8-character identifier (hash of source+dataset); fetch directly with `eosbench fetch --id <id>` |
| `dataset` | family name, e.g. `ames`, `tox21` (the underlying field is `name`; sort with `--sort_by name`) |
| `source` | registry the data comes from, e.g. `tdcommons`, `moleculenet`, `polaris` |
| `task` | `cls` (classification) or `reg` (regression) |
| `columns` | number of label columns (endpoints) sharing this family's molecules and split; `1` for single-label sets |
| `n_tot` | total samples (molecule count, for multi-column families) |
| `size` | full on-disk footprint including the feature matrices, e.g. `126 MB` |
| `n_pos` | positive-class samples — classification only, blank for regression |
| `balance` | task-aware label-shape cue: a class-balance bar + `n_pos/n_tot` ratio for classification, or a skewness bar for regression (unbounded, saturates past \|skew\|≥2) |
| `baseline` | RandomForest baseline averaged over random K-fold CV — `AUROC/AUPRC` for classification, `RMSE/R²` for regression. A reference floor for "how hard is this dataset," **not** the best published model |
| `leaderboard` | best *published* result and its metric where known, e.g. `0.871 AUROC ±`; blank when no reference exists. The trailing glyph is the comparability signal explained below |
| `lb_split` | the split that published `leaderboard` score was computed on: `scaffold`, `random`, or `external` (evaluated on a genuinely different validation set, not a partition of this dataset at all) — not necessarily this row's own split; see the glyph below for whether it actually matches |
| `lb_provider` | where the `leaderboard` score came from: `tdc` (TDC's own ADMET Benchmark Group leaderboard, on `tdcommons` rows), `polaris` (Polaris Hub's own leaderboard, on `polaris` rows), `moleculenet` (MoleculeNet's own leaderboard — on its own rows, or cross-filled onto a `tdcommons` row covering the same assay), or `literature` for a single-paper reference |
| `last_updated` | date this family's metadata was last (re)built, `YYYY-MM-DD` |

> **Can I directly compare my own result to the `leaderboard` number in this row?** Not
> always — even when it's the same test set. The trailing glyph on `leaderboard` answers
> this directly, per row:
>
> | glyph | meaning |
> |-------|---------|
> | ✓ `yes` | same test set as this row's own scaffold holdout, **and** the published score is a single evaluation, same as eosbench's own `scaffold_auroc` (one model, fit once on train, scored once on test) — directly comparable |
> | ± `split_only` | same test set, but the published score is itself a **multi-run average** (e.g. TDC requires 5 seeded runs, averaged ± std) — a different *kind* of number, not just a different draw. A single run of yours can land above or below a 5-run mean by chance even on identical data |
> | ✗ `no` | a genuinely different test set/split — cross-filled from another dataset's own copy (e.g. MoleculeNet's), or a paper's own ad hoc split |
> | ? `unverified` | not independently checked either way — don't assume it matches |
>
> Why this needed spelling out — the actual trail, since it wasn't obvious at first:
>
> - **`tdcommons/ames`** → `leaderboard: 0.871 AUROC ±`, `lb_split: scaffold`, `lb_provider:
>   tdc`. That `0.871` is ZairaChem's entry on **TDC's own** ADMET Benchmark Group
>   leaderboard (`tdcommons.ai`) — not something read off Polaris Hub, despite this project
>   once assuming otherwise (`lb_provider` used to say `polaris` here; fixed after checking
>   TDC's leaderboard page directly, where ZairaChem's `0.871 ± 0.002` matches exactly, and
>   confirming ZairaChem never appears among Polaris Hub's own submitted results for this
>   benchmark). TDC's leaderboard **requires 5 independent training runs** (different seeded
>   train/valid splits, same frozen test set each time) and reports the mean ± std across
>   them — so `0.871` is a 5-run average, not a single evaluation. The test set itself *is*
>   the same one eosbench's own `tdcommons/ames` (and `polaris/ames`) scaffold holdout uses —
>   verified byte-for-byte, every molecule and split assignment identical — hence `±`, not
>   `✗`: same data, different statistic.
> - **`polaris/ames`** (the same underlying dataset, prepared directly from Polaris rather
>   than PyTDC) → `leaderboard: 0.877 AUROC ✓`, `lb_split: scaffold`, `lb_provider: polaris`.
>   This is a genuinely different, separate leaderboard — Polaris Hub's own live results for
>   its `tdcommons/ames` benchmark artifact, fetched by `fetch_polaris_leaderboard.py` (see
>   `leaderboard_fetched_at`). Its submitter roster doesn't overlap with TDC's leaderboard at
>   all (no ZairaChem, no MiniMol — different models entirely). Polaris's own result schema
>   has no field for a run count or std at all: a submission is always one reported number,
>   same as eosbench's single-run `scaffold_auroc` — hence `✓`, on the same (verified
>   identical) test set as the `tdcommons` row above.
> - **`tdcommons/clintox`** → `leaderboard: 0.832 AUROC ✗`, `lb_split: random`, `lb_provider:
>   moleculenet`. ClinTox also exists as a MoleculeNet benchmark, so eosbench cross-fills
>   MoleculeNet's best published score onto this row — but that's *MoleculeNet's own* split
>   on *MoleculeNet's own* copy of the data, not this row's. Same assay, different partition.
> - **`moleculenet/bace`** → `leaderboard: 0.806 AUROC ?`. This is BACE's *own* source citing
>   its *own* paper's leaderboard number — but eosbench computes its own scaffold split for
>   MoleculeNet families rather than honouring a frozen official one (unlike
>   tdcommons/polaris), and whether that matches Wu et al.'s original split hasn't been
>   independently checked. `?`, not `✓`, until it is.
>
> The metric itself also varies row to row — most are `AUROC`, but e.g.
> `tdcommons/cyp2c9_substrate_carbonmangels` reports `0.474 AUPRC`, because that's the metric
> its own reference used. Compare `leaderboard` only within a row, never across rows.

`--expand` swaps `columns`/`n_pos` for one row per label `column` instead of per family.
The footer (`N families · M label columns`) counts across every row shown.

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
┌────────────────────────┬───────────────────────────────────────────────┐
│ id                     │ bed0959b                                     │
│ source                 │ tdcommons                                    │
│ dataset                │ ames                                         │
│ task                   │ classification                               │
│ n_molecules            │ 7278                                         │
│ n_columns              │ 1                                            │
│ leaderboard            │ 0.8710 ± 0.0020 (AUROC)                      │
│ leaderboard_split      │ scaffold                                     │
│ leaderboard_provider   │ tdc                                          │
│ leaderboard_comparable │ split_only                                   │
│ leaderboard_source     │ ZairaChem, TDC ADMET Benchmark Group         │
│                        │ leaderboard (5-run avg)                      │
│ last_updated           │ 2026-08-03                                   │
└────────────────────────┴───────────────────────────────────────────────┘
                             columns
┏━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━ … ━┓
┃ column ┃ n    ┃ pos  ┃ neg  ┃ auroc (random)  ┃ auprc (random)  ┃     ┃
┡━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━ … ━┩
│ ames   │ 7278 │ 3974 │ 3304 │ 0.9090 ± 0.0070 │ 0.9199 ± 0.0067 │ …   │
└────────┴──────┴──────┴──────┴─────────────────┴─────────────────┴─ … ─┘
auroc/auprc are a RandomForest baseline (a reference floor, not the best
published model). 'random split' = mean over random K-fold cross-validation;
'scaffold split' = a single Bemis–Murcko scaffold holdout.
```

`leaderboard_comparable` is the same signal shown as a glyph in `eosbench catalog` —
here it's spelled out (`split_only`: TDC's `0.871` is a 5-run average on the same test
set eosbench's own `scaffold_auroc` uses, not a single evaluation like eosbench's own —
see [Leaderboard references](#leaderboard-references)). Running the same command with
`--source polaris --dataset ames` instead shows a *different*, genuinely-live leaderboard
number (`leaderboard_comparable: yes`, `leaderboard_fetched_at: 2026-08-06`) for the
identical underlying molecules — two separate leaderboards, not two views of one.

Every dataset is a **family**, so `info` shows a summary plus a per-column table
(`tdcommons`/`polaris` ADMET tasks and most `moleculenet` sets are 1-column families).

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
eosbench fetch --id bed0959b                       # simplest: fetch by identifier
eosbench fetch --source tdcommons --dataset ames    # or explicitly
eosbench fetch --source tdcommons --dataset ames --featurization rdkit --output_dir my_data
eosbench fetch --source moleculenet --dataset bbbp --featurization none
# copy from a locally prepared mirror instead of S3 (e.g. before uploading)
eosbench fetch --source tdcommons --dataset ames --from_dir data
```

| argument | default | description |
|----------|---------|-------------|
| `--id` | — | eosbench identifier; resolves source/dataset/task automatically (use instead of `--source`/`--dataset`) |
| `--source` | required* | `tdcommons`, `moleculenet`, or `polaris` (*not needed with `--id`) |
| `--dataset` | required* | dataset name, e.g. `ames` (*not needed with `--id`) |
| `--featurization` | `morgan` | `morgan`, `rdkit`, or `none` |
| `--output_dir` | `.` | root folder to write into |
| `--task` | `classification` | `classification` or `regression` |
| `--from_dir` | `None` | copy from a local mirror laid out as `DIR/{source}/{task}/{dataset}/` instead of downloading from S3 |

---

## Preparing new datasets

New datasets are built with the scripts under `scripts/`. These require extra tooling
(rdkit, scikit-learn, a source-specific client) that is **not** needed to consume eosbench.
Installation is split into **three separate extras**, one per source, rather than one
combined `[prepare]` extra:

```bash
pip install -e ".[prepare-moleculenet]"
pip install -e ".[prepare-tdcommons]"
pip install -e ".[prepare-polaris]"
```

They're split because `PyTDC`'s dependency tree and `polaris-lib` want incompatible versions
of the AWS SDK (`boto3`) — installing both together in one environment makes pip's resolver
backtrack into a broken, unrelated `boto3` version. `prepare-tdcommons` also pins
`rdkit==2023.9.6` and `PyTDC==0.4.17` specifically, not just "latest" — see the comments next
to that extra in `pyproject.toml` for why those exact versions matter for reproducibility.

To add MoleculeNet datasets:

```bash
pip install -e ".[prepare-moleculenet]"
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

To add Polaris datasets:

```bash
pip install -e ".[prepare-polaris]"
python scripts/prepare_polaris.py                          # auto-discover all qualifying benchmarks
python scripts/prepare_polaris.py --datasets tdcommons/ames
python scripts/prepare_polaris.py --limit 5 --no_baseline  # quick pass
```

No Polaris Hub login is needed for this — listing benchmarks and loading their datasets/splits
are public, unauthenticated reads (a login is only needed for *writing* results back to the
Hub, which nothing here does). This enumerates Polaris Hub benchmarks and keeps the
**single-input, binary-classification** ones (multi-input, regression, and multiclass
benchmarks are skipped, each logged with a reason). Single-target benchmarks become 1-column
families; single-input multi-target benchmarks become multi-column families. Polaris hides
test labels only behind its split API — the labels live in the underlying dataset table, which
the script reads directly — and unlike MoleculeNet the `scaffold_split` column carries
Polaris's **official** train/test split (`scaffold_split_method: "polaris"`) rather than a
computed scaffold split; a random K-fold is added alongside.

In practice, the Hub's `tdcommons` owner mirrors the entire **TDC ADMET Benchmark Group** — 22
benchmarks total (verified by listing them directly), 13 single-input binary-classification
(the ones eosbench currently prepares — see [Datasets](#datasets)) and 9 regression (not yet
prepared here). For each of those 13, Polaris's own official split is **verified
byte-for-byte identical** to the frozen split `tdcommons`'s own `prepare_tdcommons.py` honours
for the same dataset (same molecules, same train/test assignment) — the two sources end up
describing the same underlying test set through two different pipelines. What they do *not*
share is a leaderboard: see [Leaderboard references](#leaderboard-references) for why Polaris's
own live leaderboard for these benchmarks is a completely separate thing from TDC's.

Unlike `prepare_tdcommons.py`/`prepare_moleculenet.py`, `prepare_polaris.py` itself attaches
**no** leaderboard reference at prep time (it always passes `leaderboard=None`) — run
`scripts/fetch_polaris_leaderboard.py` afterwards to fetch and attach Polaris's own live
leaderboard scores (it re-syncs `metadata.json` in the same run, no separate patch step
needed).

To add TDC (Therapeutics Data Commons) datasets:

```bash
pip install -e ".[prepare-tdcommons]"
python scripts/prepare_tdcommons.py                       # all binary ADMET datasets
python scripts/prepare_tdcommons.py --datasets ames,herg  # specific datasets
python scripts/prepare_tdcommons.py --limit 5 --no_baseline
```

This auto-discovers the single-prediction **ADME**, **Tox** and **HTS** (high-throughput
screening bioassay) datasets via `PyTDC`, keeps the **binary-classification** ones (regression
datasets are skipped, each logged), and writes one family per dataset under the
**`tdcommons`** source. Most are 1-column families; a handful (`tox21`, `toxcast`,
`herg_central`) are multi-label on TDC — each label is served as a *separate* API call with
its own molecule subset, so those are aligned by `Drug_ID` into one conserved multi-column
family instead. Sizes range from ~880 molecules to ~340k, giving benchmarks of varied scale.
Other `single_pred` groups are excluded on purpose (QM/Yields are regression;
Epitope/Paratope/Develop/CRISPROutcome take protein/sequence inputs, not SMILES).

Single-label datasets honour **TDC's own scaffold split** as the holdout
(`scaffold_split_method: "tdc-scaffold"`), generated with a fixed seed (`42`, kept as its own
constant independent of `--seed`) — verified empirically to reproduce the TDC ADMET Benchmark
Group's frozen leaderboard test set byte-for-byte (exact `Drug_ID` match) for every one of its
22 members. Multi-label families have no such split available (no single TDC split covers the
assembled union) and get eosbench's own computed Murcko scaffold split instead
(`scaffold_split_method: "murcko"`), like MoleculeNet's multi-column families — with no seed
at all, since it's fully deterministic given the molecule set. A random K-fold CV is added
alongside either way.

Leaderboard references for `tdcommons` datasets are sourced with a clear precedence — **TDC's
own** ADMET Benchmark Group leaderboard first, then a same-assay **MoleculeNet** cross-fill,
then a single **literature** reference as a last resort. The `polaris` source instead carries
Polaris Hub's own, separately-fetched leaderboard. See
[Leaderboard references](#leaderboard-references) below for the full picture, including which
of these a local result is actually comparable to.

Files are written to `data/{source}/classification/{family}/` (one multi-column `data.csv`,
one `folds.csv`, one feature matrix per featurizer) and a copy of `metadata.json` is bundled
under `src/eosbench/_data/...` so the family shows up in the catalog. The heavy files are
**not** uploaded automatically — publish them with Ersilia's internal `eosvc` tool, scoped to
what you just prepared:

```bash
eosvc upload --path data/{source}/classification/{family}
```

(avoid a blanket `--path data/` — raw prep caches such as `data/_raw/`/`.tdc_raw_cache/` don't
belong on the public bucket).

The shared core lives in `scripts/_prepare_common.py` so additional sources can be added
as `scripts/prepare_<source>.py` that reuse `prepare_family(...)`. Sources that ship their own
train/test split (like Polaris) pass it via `prepare_family(..., holdout=..., holdout_method=...)`.

### Leaderboard references

The `leaderboard_score`/`leaderboard_metric`/`leaderboard_split`/`leaderboard_provider`/
`leaderboard_comparable` columns record the best **published** result per dataset (distinct
from the RandomForest `baseline`). `leaderboard_std` — the reported cross-seed standard
deviation, where known — and `leaderboard_fetched_at` — when a *live-fetched* score was
captured — are also recorded in `metadata.json` (not catalog columns). These come from
curated JSONs under `scripts/`:

- **MoleculeNet** — `scripts/moleculenet_leaderboard.json`, applied at prep time by
  `prepare_moleculenet.py`. **Manual only** — there's no live-fetch script for this one, unlike
  the two below: `moleculenet.org` currently returns HTTP 404 on both its root and dataset
  pages (checked 2026-08-06 — looks rebuilt/repurposed, not the original DeepChem site), so
  there's nothing to scrape even if it were worth automating. It wouldn't be, either: this
  "leaderboard" is a fixed set of best-model numbers from one 2018 paper, not a live, growing
  one like the two below — nothing new would show up on a re-fetch even if the site came back.
  Update the JSON by hand (new paper, better score) if needed. Every entry is `provider:
  moleculenet` (the dataset's own paper) and `comparable: unverified` — eosbench computes its
  own scaffold/random split for MoleculeNet families rather than honouring an official frozen
  one, and that hasn't been independently checked against the original paper's split.
- **tdcommons** — `scripts/tdcommons_leaderboard.json`, applied at prep time by
  `prepare_tdcommons.py`. Mixed provenance — three of its `provider` values are hand-curated
  and only patchable after the fact (`scripts/patch_leaderboard_metadata.py`), but the fourth
  is now genuinely live-fetched:
  - `tdc` — **TDC's own** ADMET Benchmark Group leaderboard (`tdcommons.ai`), a 5-independent-run
    average per TDC's own submission guide; the 13 ADMET classification tasks. Live-fetched by
    `scripts/fetch_tdc_leaderboard.py` (re-run to refresh; syncs `metadata.json` in the same
    run) — TDC's leaderboard gains real new submissions over time via its own Google-Form
    process, unlike MoleculeNet's. (Previously mislabeled `polaris` here, on the incorrect
    assumption that Polaris Hub mirrors this leaderboard — checked directly against
    `tdcommons.ai`'s own per-dataset pages and against Polaris Hub's actual submitted results
    on 2026-08-06: no overlap in model names at all.) `comparable: split_only` — same frozen
    test set as this row's own scaffold holdout (verified byte-for-byte), but a multi-run
    average, not a single evaluation. `fetch_tdc_leaderboard.py` only ever touches these 13
    entries (the ones it finds classified `Binary` on TDC's own ADMET Benchmark Group pages);
    the `moleculenet`/`literature` entries below aren't part of that group at all, so they're
    untouched by every run and stay purely hand-curated.
  - `moleculenet` — for datasets that are the *same* as a MoleculeNet benchmark (`hiv`,
    `clintox`); hand-curated, same caveats as MoleculeNet's own entries above.
    `comparable: no` — MoleculeNet's own copy/split, not this row's.
  - `literature` — a single reputable/recent paper, for datasets on no leaderboard at all
    (`cyp1a2_veith`, `cyp2c19_veith`, `b3db_classification`, `herg_karim`, `hlm`, `rlm`);
    hand-curated. **These are references on each paper's own split/metric — not comparable**
    (`comparable: no`) to one another, to the TDC numbers, or to eosbench's baseline; the
    `split` and `source` fields record the provenance.
- **polaris** — `scripts/polaris_leaderboard.json`, maintained by `scripts/fetch_polaris_leaderboard.py`
  (re-run it to refresh; it also re-syncs `metadata.json` in the same run). This is the
  genuinely-live Polaris Hub leaderboard for its own `tdcommons/*` benchmark artifacts — a
  **different, separately-populated leaderboard** from TDC's own (no overlap in submitters),
  fetched by parsing the public benchmark page's embedded results (there's no anonymous,
  documented API for this — see the script's docstring; `fetch_tdc_leaderboard.py` needs no
  such trick, since TDC's own pages are plain server-rendered HTML tables). Always `provider:
  polaris`, `comparable: yes` — Polaris's result schema has no run-count/std field at all, so a
  submission is always a single number, same statistical kind as eosbench's own
  `scaffold_auroc`, on the same test set as the matching `tdcommons` row.

  Datasets with no clean dataset-specific number stay blank (the Butkiewicz HTS panel is
  reported as logAUC rather than ROC-AUC; SARS-CoV-2, PAMPA, `skin_reaction`,
  `carcinogens_lagunin`, and multi-label `tox21` have no comparable published AUROC).

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
`main` (and can be triggered manually from the Actions tab). Because the availability badges are
checked at build time, re-run the workflow after uploading new datasets to S3 to refresh them.
To preview locally:

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
