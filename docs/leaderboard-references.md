# Leaderboard references

The `leaderboard_score`/`leaderboard_metric`/`leaderboard_split`/`leaderboard_provider`/
`leaderboard_comparable` columns record the best published result per dataset (distinct from
the RandomForest `baseline`). `leaderboard_std` (the reported cross-seed standard deviation,
where known) and `leaderboard_fetched_at` (when a live-fetched score was captured) are also
recorded in `metadata.json`, but aren't catalog columns. These come from curated JSONs under
`scripts/`:

- **MoleculeNet** — `scripts/moleculenet_leaderboard.json`, applied at prep time by
  `prepare_moleculenet.py`. Manual only — there's no live-fetch script for this one. As of
  2026-08-06, `moleculenet.org` returns HTTP 404 on both its root and dataset pages, so there's
  nothing to scrape even if it were worth automating. It wouldn't be, either: this "leaderboard"
  is a fixed set of best-model numbers from one 2018 paper, not a live, growing one — nothing
  new would show up on a re-fetch even if the site came back. Update the JSON by hand if a
  better score turns up. Every entry is `provider: moleculenet` and `comparable: unverified` —
  eosbench computes its own scaffold/random split for MoleculeNet families rather than
  honouring an official frozen one, and that hasn't been checked against the original paper's
  split.
- **tdcommons** — `scripts/tdcommons_leaderboard.json`, applied at prep time by
  `prepare_tdcommons.py`. Mixed provenance: two of its providers are hand-curated, the third
  is live-fetched.
  - `tdc` — TDC's own ADMET Benchmark Group leaderboard (`tdcommons.ai`), a 5-independent-run
    average per TDC's own submission guide, for the 13 ADMET classification tasks. Live-fetched
    by `scripts/fetch_tdc_leaderboard.py` (re-run to refresh; syncs `metadata.json` in the same
    run), since TDC's leaderboard gains real new submissions over time via its own Google-Form
    process. `comparable: split_only` — same frozen test set as this row's own scaffold holdout
    (verified byte-for-byte), but a multi-run average, not a single evaluation.
    `fetch_tdc_leaderboard.py` only ever touches these 13 entries — the `moleculenet`/
    `literature` entries below aren't part of the ADMET Benchmark Group at all, so they stay
    purely hand-curated.
  - `moleculenet` — for datasets that are the same as a MoleculeNet benchmark (`hiv`,
    `clintox`); hand-curated. `comparable: no` — MoleculeNet's own copy/split, not this row's.
  - `literature` — a single reputable/recent paper, for datasets on no leaderboard at all
    (`cyp1a2_veith`, `cyp2c19_veith`, `b3db_classification`, `herg_karim`, `hlm`, `rlm`);
    hand-curated. These are references on each paper's own split/metric, not comparable to one
    another, to the TDC numbers, or to eosbench's baseline (`comparable: no`); `split`/`source`
    record the provenance.
- **polaris** — `scripts/polaris_leaderboard.json`, maintained by
  `scripts/fetch_polaris_leaderboard.py` (re-run to refresh; also re-syncs `metadata.json`).
  This is the genuinely-live Polaris Hub leaderboard for its own `tdcommons/*` benchmark
  artifacts — a different, separately-populated leaderboard from TDC's own, with no overlap in
  submitters. Fetched by parsing the public benchmark page's embedded results, since there's no
  anonymous, documented API for reading Hub leaderboard scores (`fetch_tdc_leaderboard.py`
  needs no such trick — TDC's own pages are plain server-rendered HTML tables). Always
  `provider: polaris`, `comparable: yes` — Polaris's result schema has no run-count/std field
  at all, so a submission is always a single number, the same statistical kind as eosbench's
  own `scaffold_auroc`, on the same test set as the matching `tdcommons` row.

Datasets with no clean dataset-specific number stay blank (the Butkiewicz HTS panel is reported
as logAUC rather than ROC-AUC; SARS-CoV-2, PAMPA, `skin_reaction`, `carcinogens_lagunin`, and
multi-label `tox21` have no comparable published AUROC).

## Why `tdcommons` and `polaris` are two separate leaderboards, not one

Polaris Hub's `tdcommons` owner mirrors the entire TDC ADMET Benchmark Group's *datasets and
splits* (22 benchmarks, 13 classification + 9 regression) — Polaris's official split for each
of the 13 classification tasks is verified byte-for-byte identical to the frozen split
`prepare_tdcommons.py` honours for the same dataset. What Polaris does **not** mirror is TDC's
*leaderboard*. That was this project's original assumption, and it was wrong: `provider: polaris`
used to be recorded for `tdcommons`'s 13 ADMET entries, on the belief that the Polaris Hub
mirrors TDC's own leaderboard. Checked directly on 2026-08-06 against both platforms:

- **`tdcommons.ai`'s own leaderboard page** for `ames` lists ZairaChem first at
  `0.871 ± 0.002` — matching the curated number exactly.
- **Polaris Hub's own submitted results** for its `tdcommons/ames` benchmark list a completely
  different roster (`next-size-adjust`, `CheMeleon`, `TabPFNv2`, …) — no ZairaChem, no MiniMol,
  no overlap in model names at all.

So the number was always TDC's own, never Polaris Hub's. The field is now `provider: tdc`, and
`polaris_leaderboard.json` carries Polaris Hub's own, separately-fetched numbers for the same
13 datasets under the `polaris` source — two real leaderboards for the same test set, not one
leaderboard under two names.

That distinction also explains why `tdcommons/ames` shows `comparable: split_only` while
`polaris/ames` shows `comparable: yes` for the identical molecules: TDC's leaderboard reports a
5-independent-run average (its own submission guide requires 5 seeded runs, scored on the same
frozen test set, then averaged), while Polaris Hub's result schema has no run-count or std field
at all — a submission there is always a single number, the same statistical kind as eosbench's
own single-run `scaffold_auroc`. Same test set, different kind of number.
