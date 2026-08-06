# Preparing new datasets

New datasets are built with the scripts under `scripts/`. These need extra tooling (rdkit,
scikit-learn, a source-specific client) that isn't required to consume eosbench. Installation
is split into three separate extras, one per source, rather than one combined `[prepare]`
extra:

```bash
pip install -e ".[prepare-moleculenet]"
pip install -e ".[prepare-tdcommons]"
pip install -e ".[prepare-polaris]"
```

They're split because `PyTDC`'s dependency tree and `polaris-lib` want incompatible versions
of the AWS SDK (`boto3`) — installing both together makes pip's resolver backtrack into a
broken, unrelated `boto3` version. `prepare-tdcommons` also pins `rdkit==2023.9.6` and
`PyTDC==0.4.17` specifically, not just "latest" — see the comments next to that extra in
`pyproject.toml` for why those exact versions matter for reproducibility.

## MoleculeNet

```bash
pip install -e ".[prepare-moleculenet]"
python scripts/prepare_moleculenet.py                      # default subset
python scripts/prepare_moleculenet.py --datasets bbbp,tox21
python scripts/prepare_moleculenet.py --datasets all --n_folds 5 --seed 42
```

For each set this writes one family:

- downloads the raw CSV from the public DeepChem mirror;
- keeps every binary endpoint as a label column (NaN where unlabeled); single-column sets
  become 1-column families;
- computes `morgan` and `rdkit` features once for the shared molecules;
- writes a single conserved split into `folds.csv` — a random K-fold (`random_fold`) and a
  deterministic Bemis–Murcko scaffold holdout (`scaffold_split`) shared by all columns (whole
  scaffold groups stay together, so train and test never share a scaffold). Single-column
  families get a class-stratified split so every fold and the scaffold test set contain both
  classes (a naive scaffold split puts rare singleton scaffolds in the test tail, which on
  skewed sets like BBBP gives a single-class test set with undefined AUROC). Multi-column
  families have no single label to stratify on, so they use a plain conserved split, and the
  baseline skips any column whose restricted test set is single-class;
- trains a per-column RandomForest baseline and records both AUROC and AUPR (random-CV mean ±
  std and the scaffold-holdout score) per column, plus the published leaderboard reference and
  a `last_updated` date. Use `--no_baseline` to skip the baseline (faster for large families
  like ToxCast/MUV).

## Polaris

```bash
pip install -e ".[prepare-polaris]"
python scripts/prepare_polaris.py                          # auto-discover all qualifying benchmarks
python scripts/prepare_polaris.py --datasets tdcommons/ames
python scripts/prepare_polaris.py --limit 5 --no_baseline  # quick pass
```

No Polaris Hub login is needed: listing benchmarks and loading their datasets/splits are
public, unauthenticated reads (a login is only needed to *write* results back to the Hub, which
nothing here does). This enumerates Polaris Hub benchmarks and keeps the single-input,
binary-classification ones (multi-input, regression, and multiclass benchmarks are skipped,
each logged with a reason). Single-target benchmarks become 1-column families; single-input
multi-target benchmarks become multi-column families. Polaris hides test labels only behind its
split API — the labels live in the underlying dataset table, which the script reads directly —
and unlike MoleculeNet, `scaffold_split` carries Polaris's official train/test split
(`scaffold_split_method: "polaris"`) rather than a computed one; a random K-fold is added
alongside.

In practice, the Hub's `tdcommons` owner mirrors the entire TDC ADMET Benchmark Group — 22
benchmarks, 13 single-input binary-classification (the ones eosbench currently prepares) and 9
regression (not yet prepared here). For each of the 13, Polaris's own official split is
verified byte-for-byte identical to the frozen split `prepare_tdcommons.py` honours for the
same dataset — the two sources describe the same underlying test set through two different
pipelines. What they don't share is a leaderboard: see
[leaderboard-references.md](leaderboard-references.md) for why Polaris's own live leaderboard
for these benchmarks is a separate thing from TDC's.

Unlike `prepare_tdcommons.py`/`prepare_moleculenet.py`, `prepare_polaris.py` itself attaches no
leaderboard reference at prep time — run `scripts/fetch_polaris_leaderboard.py` afterwards to
fetch and attach Polaris's own live leaderboard scores (it re-syncs `metadata.json` in the same
run, no separate patch step needed).

## TDC (Therapeutics Data Commons)

```bash
pip install -e ".[prepare-tdcommons]"
python scripts/prepare_tdcommons.py                       # all binary ADMET datasets
python scripts/prepare_tdcommons.py --datasets ames,herg  # specific datasets
python scripts/prepare_tdcommons.py --limit 5 --no_baseline
```

This auto-discovers the single-prediction ADME, Tox and HTS (high-throughput screening
bioassay) datasets via `PyTDC`, keeps the binary-classification ones (regression datasets are
skipped, each logged), and writes one family per dataset under the `tdcommons` source. Most are
1-column families; a handful (`tox21`, `toxcast`, `herg_central`) are multi-label on TDC — each
label is served as a separate API call with its own molecule subset, so those are aligned by
`Drug_ID` into one conserved multi-column family instead. Sizes range from ~880 molecules to
~340k. Other `single_pred` groups are excluded on purpose (QM/Yields are regression;
Epitope/Paratope/Develop/CRISPROutcome take protein/sequence inputs, not SMILES).

Single-label datasets honour TDC's own scaffold split as the holdout
(`scaffold_split_method: "tdc-scaffold"`), generated with a fixed seed (`42`, kept as its own
constant independent of `--seed`) — verified empirically to reproduce the TDC ADMET Benchmark
Group's frozen leaderboard test set byte-for-byte (exact `Drug_ID` match) for all 22 members.
Multi-label families have no such split available and get eosbench's own computed Murcko
scaffold split instead (`scaffold_split_method: "murcko"`), like MoleculeNet's multi-column
families, with no seed at all since it's fully deterministic given the molecule set. A random
K-fold CV is added alongside either way.

Leaderboard references for `tdcommons` datasets are sourced with a clear precedence — TDC's own
ADMET Benchmark Group leaderboard first, then a same-assay MoleculeNet cross-fill, then a
single literature reference as a last resort. The `polaris` source instead carries Polaris
Hub's own, separately-fetched leaderboard. See
[leaderboard-references.md](leaderboard-references.md) for the full picture, including which
of these a local result is actually comparable to.

## Publishing

Files are written to `data/{source}/classification/{family}/` (one multi-column `data.csv`,
one `folds.csv`, one feature matrix per featurizer), and a copy of `metadata.json` is bundled
under `src/eosbench/_data/...` so the family shows up in the catalog. The heavy files aren't
uploaded automatically — publish them with Ersilia's internal `eosvc` tool, scoped to what you
just prepared:

```bash
eosvc upload --path data/{source}/classification/{family}
```

Avoid a blanket `--path data/` — raw prep caches such as `data/_raw/`/`.tdc_raw_cache/` don't
belong on the public bucket.

## Adding a new source

The shared core lives in `scripts/_prepare_common.py`, so additional sources can be added as
`scripts/prepare_<source>.py` reusing `prepare_family(...)`. A source that ships its own
train/test split (like Polaris) passes it via
`prepare_family(..., holdout=..., holdout_method=...)`.
