# Ersilia benchmarks for ML training

> **Work in progress:** this repository is under active development. Datasets, APIs, and
> documentation may change without notice.

A Python package for loading molecular activity datasets used in Ersilia ML development.

> **Important:** this is not an official benchmark. The `baseline` metric (random K-fold CV)
> uses an arbitrary split and isn't comparable to outside results. The scaffold-holdout metric
> honours the source's own official split for `tdcommons`/`polaris` datasets, but whether a
> given published `leaderboard` number is a fair comparison to it varies row by row — see
> `eosbench catalog`'s `leaderboard` glyph, or [docs/leaderboard-references.md](docs/leaderboard-references.md).

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

The typical use case is benchmarking a machine learning model across multiple datasets:

1. Browse available datasets:
   ```bash
   eosbench catalog
   ```

2. Fetch the ones you want:
   ```bash
   cd my_project/
   eosbench fetch --source tdcommons --dataset ames --featurization morgan
   eosbench fetch --source tdcommons --dataset herg --featurization morgan
   ```
   This downloads `data.csv`, `folds.csv`, `morgan.npy`, and `metadata.json` into
   `./tdcommons/classification/ames/` (and `herg/`). Files already present aren't
   re-downloaded.

3. Load and evaluate in Python:
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

`load_dataset` uses the same local cache (`~/.cache/eosbench/`) as `eosbench fetch`, so
already-downloaded files are reused automatically.

---

## Datasets

Browse everything online at [ersilia-os.github.io/eosbench](https://ersilia-os.github.io/eosbench/)
(see [docs/catalog-website.md](docs/catalog-website.md)). `eosbench` ships metadata for three
sources:

- **tdcommons** — single-input SMILES binary-classification datasets from
  [Therapeutics Data Commons](https://tdcommons.ai/): ADMET properties (Ames, hERG, BBB,
  CYP450, …) plus HTS bioassays (SARS-CoV-2, Butkiewicz panel, HIV). Built by
  `scripts/prepare_tdcommons.py`.
- **MoleculeNet** — classification: BBBP, BACE, HIV, Tox21, ClinTox, SIDER, MUV, ToxCast.
  Regression: ESOL, FreeSolv, Lipophilicity, QM8/QM9.
- **polaris** — the 13 ADMET classification benchmarks Polaris Hub mirrors from TDC, prepared
  directly from Polaris instead of PyTDC. A deliberate second copy of the same 13 datasets: it
  carries Polaris Hub's own live leaderboard, which is a different, separately-populated thing
  from TDC's own. See [docs/preparing-datasets.md](docs/preparing-datasets.md).

Datasets are organized as **families**: a family is one or more binary label **columns**
(endpoints) over a shared set of molecules, sharing the same feature matrices and a single
conserved train/test split. The vocabulary end to end: **source** (`tdcommons`,
`moleculenet`, `polaris`) → **dataset** (the family, e.g. `tox21`) → **column** (an endpoint,
e.g. `NR-AR`); **task** is `classification`/`regression`.

Dataset files are downloaded on demand from a public S3 bucket and cached under
`~/.cache/eosbench/`.

---

## API

`eosbench` also exposes a small Python API (`get_catalog`, `load_dataset`, `list_columns`)
mirroring the CLI, used in the [How to use](#how-to-use) example above. It isn't fully
documented yet — for now, the CLI is the more complete reference.

---

## CLI

| command | description |
|---------|-------------|
| `eosbench catalog` | list available datasets, with filtering/sorting/limiting |
| `eosbench info` | show metadata for one dataset |
| `eosbench fetch` | download a dataset to a local folder |

Full reference, including every column/flag and the `leaderboard` comparability glyph:
[docs/cli.md](docs/cli.md).

---

## Preparing new datasets

New datasets are built with `scripts/prepare_<source>.py`, which need extra tooling (rdkit,
scikit-learn, a source-specific client) not required to consume eosbench:

```bash
pip install -e ".[prepare-moleculenet]"   # or prepare-tdcommons / prepare-polaris
python scripts/prepare_tdcommons.py --datasets ames,herg
```

Prepared datasets are published to the public S3 bucket with Ersilia's internal `eosvc` tool.
Full instructions, per-source details, and the leaderboard-reference pipeline:
[docs/preparing-datasets.md](docs/preparing-datasets.md) and
[docs/leaderboard-references.md](docs/leaderboard-references.md).

---

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
