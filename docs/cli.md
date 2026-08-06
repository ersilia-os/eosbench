# CLI reference

`eosbench` installs a command-line interface alongside the package:

| command | description |
|---------|-------------|
| `eosbench catalog` | list available datasets, with filtering/sorting/limiting |
| `eosbench info` | show metadata for one dataset |
| `eosbench fetch` | download a dataset to a local folder |

## `eosbench catalog`

Prints a table of all available datasets. The table adapts to the task, and counts
are shown as grouped integers (e.g. `12,665`).

```bash
eosbench catalog                                  # all sources
eosbench catalog --source tdcommons               # filter by source
eosbench catalog --task regression --sort-by rmse # regression sets, lowest RMSE first
```

### Columns

| column | meaning |
|--------|---------|
| `id` | deterministic 8-character identifier (hash of source+dataset); fetch directly with `eosbench fetch --id <id>` |
| `dataset` | family name, e.g. `ames`, `tox21` (the underlying field is `name`; sort with `--sort-by name`) |
| `source` | registry the data comes from, e.g. `tdcommons`, `moleculenet`, `polaris` |
| `task` | `cls` (classification) or `reg` (regression) |
| `columns` | number of label columns (endpoints) sharing this family's molecules and split; `1` for single-label sets |
| `n_tot` | total samples (molecule count, for multi-column families) |
| `size` | full on-disk footprint including the feature matrices, e.g. `126 MB` |
| `n_pos` | positive-class samples — classification only, blank for regression |
| `balance` | task-aware label-shape cue: a class-balance bar + `n_pos/n_tot` ratio for classification, or a skewness bar for regression (unbounded, saturates past \|skew\|≥2) |
| `baseline` | RandomForest baseline averaged over random K-fold CV — `AUROC/AUPRC` for classification, `RMSE/R²` for regression. A reference floor for how hard the dataset is, not the best published model |
| `leaderboard` | best published result and its metric where known, e.g. `0.871 AUROC ±`; blank when no reference exists. The trailing glyph is the comparability signal below |
| `lb_split` | the split the published `leaderboard` score was computed on: `scaffold`, `random`, or `external` (a genuinely different validation set, not a partition of this dataset). Not necessarily this row's own split — see the glyph |
| `lb_provider` | where the `leaderboard` score came from: `tdc` (TDC's own ADMET Benchmark Group leaderboard, on `tdcommons` rows), `polaris` (Polaris Hub's own leaderboard, on `polaris` rows), `moleculenet` (MoleculeNet's own leaderboard, on its own rows or cross-filled onto a `tdcommons` row for the same assay), or `literature` for a single-paper reference |
| `last_updated` | date this family's metadata was last (re)built, `YYYY-MM-DD` |

`--expand` swaps `columns`/`n_pos` for one row per label `column` instead of per family.
The footer (`N families · M label columns`) counts across every row shown.

### Is the `leaderboard` number comparable to my own result?

Not always, even on the same test set. The glyph after the score answers it directly:

| glyph | meaning |
|-------|---------|
| ✓ `yes` | same test set as this row's own scaffold holdout, and the published score is a single evaluation — same as eosbench's own `scaffold_auroc` (fit once, score once). Directly comparable |
| ± `split_only` | same test set, but the published score is a multi-run average (e.g. TDC requires 5 seeded runs). A different kind of number, not just a different draw of the same one |
| ✗ `no` | a genuinely different test set — cross-filled from another dataset's own copy, or a paper's own split |
| ? `unverified` | not independently checked either way |

A few real rows show the range of cases — see [leaderboard-references.md](leaderboard-references.md)
for the full story behind each, including a case where this project had it wrong (`tdcommons`'s
own `lb_provider` was mislabeled `polaris` until checked against both platforms directly):

- **`tdcommons/ames`** → `0.871 AUROC ±`. TDC's own 5-run-average leaderboard, on the same
  test set as this row's own scaffold holdout — hence `±`, not `✗`.
- **`polaris/ames`** (same underlying dataset, prepared from Polaris instead of PyTDC) →
  `0.877 AUROC ✓`. Polaris Hub's own, separately-populated leaderboard — a single-run number
  on the same test set, hence `✓`.
- **`tdcommons/clintox`** → `0.832 AUROC ✗`. Cross-filled from MoleculeNet's own copy/split
  of the same assay — a different partition entirely.
- **`moleculenet/bace`** → `0.806 AUROC ?`. BACE's own paper's number, but eosbench computes
  its own scaffold split rather than honouring an official one, and whether it matches the
  paper's split hasn't been checked.

The metric itself varies row to row too — most are `AUROC`, but e.g.
`tdcommons/cyp2c9_substrate_carbonmangels` reports `0.474 AUPRC`. Compare `leaderboard` only
within a row, never across rows.

### Filtering, sorting, limiting

- **filters** (combine with AND): `--name`, `--min/--max-samples`, `--min/--max-ratio`,
  `--min/--max-auroc`, `--min/--max-auprc` (classification), `--min/--max-rmse`,
  `--min/--max-r2` (regression). Threshold filters skip datasets with a missing value.
- **sorting/limiting**: `--sort-by COLUMN [--desc]` and `--limit N`.

```bash
eosbench catalog --min-samples 1000 --min-auroc 0.8 --sort-by auroc --desc
eosbench catalog --name cyp --limit 5
```

See `eosbench catalog --help` for the full flag and column reference.

## `eosbench info`

Shows metadata for a single dataset.

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

`leaderboard_comparable` is the same signal shown as a glyph in `eosbench catalog`, spelled
out here. Running the same command with `--source polaris --dataset ames` instead shows a
different, genuinely-live leaderboard number (`leaderboard_comparable: yes`,
`leaderboard_fetched_at: 2026-08-06`) for the identical underlying molecules — two separate
leaderboards, not two views of one; see [leaderboard-references.md](leaderboard-references.md).

Every dataset is a family, so `info` prints a summary plus a per-column table
(`tdcommons`/`polaris` ADMET tasks and most `moleculenet` sets are 1-column families). For a
multi-column family, descriptions are truncated to fit; pass `--column` for one column's full,
untruncated details:

```bash
eosbench info --source moleculenet --dataset tox21 --column NR-AhR
```

## `eosbench fetch`

Downloads a dataset to a local folder.

```bash
eosbench fetch --id bed0959b                       # simplest: fetch by identifier
eosbench fetch --source tdcommons --dataset ames    # or explicitly
eosbench fetch --source tdcommons --dataset ames --featurization rdkit --output-dir my_data
eosbench fetch --source moleculenet --dataset bbbp --featurization none
# copy from a locally prepared mirror instead of S3 (e.g. before uploading)
eosbench fetch --source tdcommons --dataset ames --from-dir data
```

| argument | default | description |
|----------|---------|-------------|
| `--id` | — | eosbench identifier; resolves source/dataset/task automatically (use instead of `--source`/`--dataset`) |
| `--source` | required* | `tdcommons`, `moleculenet`, or `polaris` (*not needed with `--id`) |
| `--dataset` | required* | dataset name, e.g. `ames` (*not needed with `--id`) |
| `--featurization` | `morgan` | `morgan`, `rdkit`, or `none` |
| `--output-dir` | `.` | root folder to write into |
| `--task` | `classification` | `classification` or `regression` |
| `--from-dir` | `None` | copy from a local mirror laid out as `DIR/{source}/{task}/{dataset}/` instead of downloading from S3 |
