import numpy as np
import pandas as pd
import pytest

from eosbench import (
    get_catalog,
    load_dataset,
    make_id,
    mirror_dataset,
)


def test_get_catalog_returns_summary_dataframe():
    catalog = get_catalog()

    assert list(catalog.columns) == [
        "id",
        "name",
        "source",
        "task",
        "n_columns",
        "n_tot",
        "size",
        "n_pos",
        "auroc",
        "auprc",
        "ratio",
        "leaderboard_score",
        "leaderboard_metric",
        "leaderboard_split",
        "leaderboard_provider",
        "leaderboard_comparable",
        "last_updated",
    ]

    # "ames" is prepared under both tdcommons (PyTDC) and polaris (Polaris's official split);
    # filter on source too so this test targets the tdcommons row specifically.
    ames = catalog.loc[
        (catalog["name"] == "ames") & (catalog["source"] == "tdcommons")
    ].iloc[0]
    assert ames["source"] == "tdcommons"
    assert ames["id"] == make_id("tdcommons", "ames")  # deterministic short code
    assert ames["n_tot"] == 7278
    assert ames["n_pos"] == 3974
    # Baseline metrics are recomputed by the prep pipeline; assert with tolerance so they
    # survive scikit-learn version drift rather than pinning an exact float.
    assert ames["auroc"] == pytest.approx(0.91, abs=0.02)
    assert ames["auprc"] == pytest.approx(0.92, abs=0.02)
    assert ames["ratio"] == pytest.approx(3974 / 7278)
    # Leaderboard reference is TDC's own ADMET Benchmark Group leaderboard (deterministic,
    # curated) -- a 5-run average on the same frozen split this row's own holdout uses.
    assert ames["leaderboard_metric"] == "AUROC"
    assert ames["leaderboard_score"] == pytest.approx(0.871)
    assert ames["leaderboard_split"] == "scaffold"
    assert ames["leaderboard_provider"] == "tdc"
    assert ames["leaderboard_comparable"] == "split_only"


def test_mirror_dataset_materializes_expected_folder_structure(tmp_path, monkeypatch):
    source = "tdcommons"
    task_type = "classification"
    dataset = "ames"
    remote_root = tmp_path / "remote"

    remote_root.mkdir()
    (remote_root / "data.csv").write_text("smiles,activity\nCCO,1\n")
    (remote_root / "folds.csv").write_text("fold\n0\n")
    np.save(remote_root / "morgan.npy", np.arange(4).reshape(1, 4))

    def fake_download(src, task, name, filename, dest):
        assert (src, task, name) == (source, task_type, dataset)
        src_path = remote_root / filename
        pdest = type(src_path)(dest)
        pdest.write_bytes(src_path.read_bytes())
        return str(pdest)

    monkeypatch.setattr("eosbench.dataset._download_to", fake_download)

    created = mirror_dataset(
        source,
        dataset,
        featurization="morgan",
        output_dir=tmp_path / "data",
        task=task_type,
    )

    assert created == tmp_path / "data" / source / task_type / dataset
    assert (created / "data.csv").is_file()
    assert (created / "folds.csv").is_file()
    assert (created / "morgan.npy").is_file()
    assert (created / "metadata.json").is_file()


def test_mirror_dataset_rejects_unknown_featurization(tmp_path):
    with pytest.raises(ValueError, match="featurization must be None"):
        mirror_dataset(
            "tdcommons", "ames", featurization="bogus", output_dir=tmp_path / "data"
        )


def test_mirror_dataset_from_dir_copies_local_files(tmp_path):
    source, task_type, dataset = "polaris", "classification", "demo"
    # A local mirror laid out exactly as the prepare scripts write under data/.
    local = tmp_path / "local" / source / task_type / dataset
    local.mkdir(parents=True)
    (local / "data.csv").write_text("smiles,demo\nCCO,1\n")
    (local / "folds.csv").write_text("random_fold,scaffold_split\n0,train\n")
    np.save(local / "morgan.npy", np.arange(4).reshape(1, 4))
    (local / "metadata.json").write_text('{"source": "polaris", "dataset": "demo"}')

    created = mirror_dataset(
        source,
        dataset,
        featurization="morgan",
        output_dir=tmp_path / "out",
        task=task_type,
        from_dir=tmp_path / "local",
    )

    assert created == tmp_path / "out" / source / task_type / dataset
    for fn in ("data.csv", "folds.csv", "morgan.npy", "metadata.json"):
        assert (created / fn).is_file()
    # metadata.json is taken from the local mirror, not the package bundle.
    assert "polaris" in (created / "metadata.json").read_text()


def test_mirror_dataset_from_dir_skips_existing_files(tmp_path):
    source, task_type, dataset = "polaris", "classification", "demo"
    local = tmp_path / "local" / source / task_type / dataset
    local.mkdir(parents=True)
    (local / "data.csv").write_text("smiles,demo\nCCO,1\n")
    (local / "folds.csv").write_text("random_fold,scaffold_split\n0,train\n")
    (local / "metadata.json").write_text('{"source": "polaris"}')

    out = tmp_path / "out"
    dest_dir = out / source / task_type / dataset
    dest_dir.mkdir(parents=True)
    # Pre-existing files are a cache hit and must be left untouched.
    (dest_dir / "data.csv").write_text("SENTINEL")
    (dest_dir / "metadata.json").write_text("SENTINEL")

    mirror_dataset(
        source,
        dataset,
        featurization=None,
        output_dir=out,
        task=task_type,
        from_dir=tmp_path / "local",
    )

    assert (dest_dir / "data.csv").read_text() == "SENTINEL"  # not overwritten
    assert (dest_dir / "metadata.json").read_text() == "SENTINEL"  # not overwritten
    assert (
        (dest_dir / "folds.csv").read_text().startswith("random_fold")
    )  # newly copied


def test_mirror_dataset_from_dir_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.csv not found.*--from-dir"):
        mirror_dataset(
            "polaris",
            "missing",
            featurization=None,
            output_dir=tmp_path / "out",
            from_dir=tmp_path / "empty",
        )


def test_load_dataset_supports_smiles_and_feature_arrays(tmp_path, monkeypatch):
    # Self-contained legacy (flat) dataset: both _fetch and the metadata are mocked, so it
    # doesn't depend on any bundled dataset (all bundled sources are now families).
    import json

    source = "legacy"
    task_type = "classification"
    dataset = "demo"

    csv_path = tmp_path / "data.csv"
    npy_path = tmp_path / "morgan.npy"
    rdkit_path = tmp_path / "rdkit.npy"
    folds_path = tmp_path / "folds.csv"
    meta_path = tmp_path / "metadata.json"

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "activity": [1.5, 0.0, 1.0],
        }
    ).to_csv(csv_path, index=False)
    np.save(npy_path, np.arange(12).reshape(3, 4))
    np.save(rdkit_path, np.arange(12, 24).reshape(3, 4))
    pd.DataFrame({"fold": [0, 1, 0]}).to_csv(folds_path, index=False)
    meta_path.write_text(
        json.dumps({"source": source, "dataset": dataset, "task": task_type})
    )

    def fake_fetch(src, task, name, filename):
        assert (src, task, name) == (source, task_type, dataset)
        mapping = {
            "data.csv": csv_path,
            "morgan.npy": npy_path,
            "rdkit.npy": rdkit_path,
            "folds.csv": folds_path,
        }
        return str(mapping[filename])

    monkeypatch.setattr("eosbench.dataset._fetch", fake_fetch)
    monkeypatch.setattr("eosbench.dataset._pkg_data_path", lambda *a: str(meta_path))

    ds_smiles = load_dataset(source, dataset, featurization=None, task=task_type)
    ds_morgan = load_dataset(source, dataset, featurization="morgan", task=task_type)
    ds_rdkit = load_dataset(source, dataset, featurization="rdkit", task=task_type)

    assert ds_smiles.X == ["CCO", "CCN", "CCC"]
    assert ds_smiles.y.tolist() == [1.5, 0.0, 1.0]
    assert len(ds_smiles.split) > 0
    assert isinstance(ds_smiles.metadata, dict)

    assert ds_morgan.X.shape == (3, 4)
    assert ds_rdkit.X.shape == (3, 4)


def test_load_dataset_rejects_unknown_featurization():
    with pytest.raises(ValueError, match="featurization must be None"):
        load_dataset("tdcommons", "ames", featurization="bogus")


def _patch_fetch(monkeypatch, tmp_path, folds_df):
    """Wire _fetch + _pkg_data_path to a tiny self-contained legacy (flat) dataset.

    Self-contained so it doesn't depend on any bundled dataset staying legacy-format
    (all bundled sources are now multi-column families).
    """
    import json

    csv_path = tmp_path / "data.csv"
    npy_path = tmp_path / "morgan.npy"
    folds_path = tmp_path / "folds.csv"
    meta_path = tmp_path / "metadata.json"

    pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC", "CCCC"], "activity": [1, 0, 1, 0]}
    ).to_csv(csv_path, index=False)
    np.save(npy_path, np.arange(16).reshape(4, 4))
    folds_df.to_csv(folds_path, index=False)
    # Legacy metadata (no "columns" key) -> load_dataset uses the flat activity/value path.
    meta_path.write_text(
        json.dumps(
            {
                "source": "legacy",
                "dataset": "demo",
                "task": "classification",
                "n_samples": 4,
            }
        )
    )

    def fake_fetch(src, task, name, filename):
        return str(
            {"data.csv": csv_path, "morgan.npy": npy_path, "folds.csv": folds_path}[
                filename
            ]
        )

    monkeypatch.setattr("eosbench.dataset._fetch", fake_fetch)
    monkeypatch.setattr("eosbench.dataset._pkg_data_path", lambda *a: str(meta_path))


def test_load_dataset_random_and_scaffold_splits(tmp_path, monkeypatch):
    _patch_fetch(
        monkeypatch,
        tmp_path,
        pd.DataFrame(
            {
                "random_fold": [0, 1, 2, 0],
                "scaffold_split": ["train", "train", "test", "test"],
            }
        ),
    )

    # metadata + data are fully mocked by _patch_fetch, so the source/dataset names are arbitrary.
    ds_random = load_dataset("legacy", "demo", featurization="morgan", split="random")
    assert len(ds_random.split) == 3  # one leave-one-fold-out pair per unique fold

    ds_scaffold = load_dataset(
        "legacy", "demo", featurization="morgan", split="scaffold"
    )
    assert len(ds_scaffold.split) == 1  # single holdout
    train_idx, test_idx = ds_scaffold.split[0]
    assert train_idx.tolist() == [0, 1]
    assert test_idx.tolist() == [2, 3]


def test_load_dataset_random_falls_back_to_legacy_fold_column(tmp_path, monkeypatch):
    _patch_fetch(monkeypatch, tmp_path, pd.DataFrame({"fold": [0, 1, 0, 1]}))
    ds = load_dataset("legacy", "demo", featurization="morgan", split="random")
    assert len(ds.split) == 2


def test_load_dataset_scaffold_without_column_raises(tmp_path, monkeypatch):
    _patch_fetch(monkeypatch, tmp_path, pd.DataFrame({"fold": [0, 1, 0, 1]}))
    with pytest.raises(ValueError, match="no 'scaffold_split' column"):
        load_dataset("legacy", "demo", featurization="morgan", split="scaffold")


def _patch_family(monkeypatch, tmp_path):
    """Wire a synthetic 2-column family (metadata + data/folds/features)."""
    import json

    metadata = {
        "source": "moleculenet",
        "dataset": "fam",
        "task": "classification",
        "n_molecules": 4,
        "n_columns": 2,
        "columns": {
            "A": {
                "n_samples": 4,
                "n_positives": 2,
                "random_auroc_mean": 0.9,
                "random_aupr_mean": 0.9,
                "description": "Endpoint A description.",
            },
            "B": {
                "n_samples": 3,
                "n_positives": 2,
                "random_auroc_mean": 0.8,
                "random_aupr_mean": 0.8,
            },
        },
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "A": [1, 0, 1, 0],
            "B": [1.0, 0.0, np.nan, 1.0],
        }
    ).to_csv(tmp_path / "data.csv", index=False)
    np.save(tmp_path / "morgan.npy", np.arange(16).reshape(4, 4))
    pd.DataFrame(
        {
            "random_fold": [0, 1, 2, 0],
            "scaffold_split": ["train", "train", "test", "test"],
        }
    ).to_csv(tmp_path / "folds.csv", index=False)

    monkeypatch.setattr("eosbench.dataset._pkg_data_path", lambda *a: str(meta_path))

    def fake_fetch(src, task, name, filename):
        return str(tmp_path / filename)

    monkeypatch.setattr("eosbench.dataset._fetch", fake_fetch)


def _patch_regression_family(monkeypatch, tmp_path):
    """Wire a synthetic single-column regression family with float targets."""
    import json

    metadata = {
        "source": "moleculenet",
        "dataset": "esolish",
        "task": "regression",
        "n_molecules": 4,
        "n_columns": 1,
        "columns": {
            "esolish": {
                "n_samples": 4,
                "random_rmse_mean": 0.5,
                "random_r2_mean": 0.7,
                "scaffold_rmse": 0.6,
                "scaffold_r2": 0.6,
            }
        },
        "rmse_mean": 0.5,
        "r2_mean": 0.7,
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC", "CCCC"],
            "esolish": [-0.77, -3.30, -2.06, 1.42],
        }
    ).to_csv(tmp_path / "data.csv", index=False)
    np.save(tmp_path / "morgan.npy", np.arange(16).reshape(4, 4))
    pd.DataFrame(
        {
            "random_fold": [0, 1, 2, 0],
            "scaffold_split": ["train", "train", "test", "test"],
        }
    ).to_csv(tmp_path / "folds.csv", index=False)

    monkeypatch.setattr("eosbench.dataset._pkg_data_path", lambda *a: str(meta_path))
    monkeypatch.setattr(
        "eosbench.dataset._fetch",
        lambda src, task, name, filename: str(tmp_path / filename),
    )


def test_load_dataset_regression_preserves_float_targets(tmp_path, monkeypatch):
    """Regression targets must NOT be floored to int by the family-format path."""
    _patch_regression_family(monkeypatch, tmp_path)

    ds = load_dataset(
        "moleculenet", "esolish", featurization="morgan", task="regression"
    )
    assert ds.y.dtype == np.float64
    # The exact float values survive (the bug would have truncated these to ints).
    assert ds.y.tolist() == [-0.77, -3.30, -2.06, 1.42]


def test_load_dataset_family_selects_column_and_masks_unlabeled_rows(
    tmp_path, monkeypatch
):
    _patch_family(monkeypatch, tmp_path)

    ds_a = load_dataset("moleculenet", "fam", featurization="morgan", column="A")
    assert ds_a.X.shape == (4, 4)
    assert ds_a.y.tolist() == [1, 0, 1, 0]
    assert ds_a.metadata["column"] == "A"
    assert (
        ds_a.metadata["description"] == "Endpoint A description."
    )  # column block merged in

    # column B has a NaN at row 2 -> that molecule is dropped from X, y, and the split.
    ds_b = load_dataset(
        "moleculenet", "fam", featurization="morgan", column="B", split="scaffold"
    )
    assert ds_b.X.shape == (3, 4)
    assert ds_b.y.tolist() == [1, 0, 1]
    train_idx, test_idx = ds_b.split[0]
    assert train_idx.tolist() == [0, 1]  # rows 0,1 (row 2 dropped, row 3 was 'test')
    assert test_idx.tolist() == [2]


def test_load_dataset_family_requires_column_when_ambiguous(tmp_path, monkeypatch):
    _patch_family(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="multiple columns"):
        load_dataset("moleculenet", "fam", featurization="morgan", column=None)


def test_load_dataset_family_rejects_unknown_column(tmp_path, monkeypatch):
    _patch_family(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="unknown column"):
        load_dataset("moleculenet", "fam", featurization="morgan", column="Z")


def test_get_catalog_expand_has_column_column():
    expanded = get_catalog(source="tdcommons", expand=True)
    collapsed = get_catalog(source="tdcommons")
    assert "column" in expanded.columns
    # One expanded row per label column; collapsed rows are 1-per-family regardless of
    # n_columns (most tdcommons families are single-column, but e.g. tox21 has 12).
    assert len(expanded) == collapsed["n_columns"].sum()


def test_make_id_is_deterministic_and_distinct():
    assert make_id("tdcommons", "ames") == make_id("tdcommons", "ames")  # stable
    assert make_id("tdcommons", "ames") != make_id(
        "moleculenet", "ames"
    )  # source matters
    assert make_id("moleculenet", "tox21") != make_id(
        "moleculenet", "tox21", "NR-AR"
    )  # family vs column


def test_resolve_id_round_trips_family_and_column():
    from eosbench.dataset import resolve_id

    fam = resolve_id(make_id("tdcommons", "ames"))
    assert (fam["source"], fam["dataset"], fam["column"]) == ("tdcommons", "ames", None)

    col = resolve_id(make_id("moleculenet", "tox21", "NR-AR"))
    assert (col["source"], col["dataset"], col["column"]) == (
        "moleculenet",
        "tox21",
        "NR-AR",
    )

    with pytest.raises(KeyError):
        resolve_id("zzzzzzzz")


def test_get_catalog_has_id_and_size_columns():
    df = get_catalog(source="tdcommons")
    assert {"id", "size"} <= set(df.columns)
    ames = df.loc[df["name"] == "ames"].iloc[0]
    assert ames["id"] == make_id("tdcommons", "ames")
    assert ames["size"] > 0  # full on-disk footprint incl. fingerprints


def test_get_catalog_collapsed_reports_median_counts(monkeypatch):
    """A multi-column family's collapsed row summarizes n_tot/n_pos by median."""

    class FakeInfo:
        dataset = "fam"
        metadata = {
            "columns": {
                "A": {"n_samples": 10, "n_positives": 4},
                "B": {"n_samples": 20, "n_positives": 6},
                "C": {"n_samples": 30, "n_positives": 2},
            }
        }

        @property
        def columns(self):
            return list(self.metadata["columns"])

    monkeypatch.setattr("eosbench.dataset.list_sources", lambda: ["fake"])
    monkeypatch.setattr(
        "eosbench.dataset.iter_datasets",
        lambda src, task: [FakeInfo()] if src == "fake" else [],
    )

    row = get_catalog().iloc[0]
    assert row["n_columns"] == 3
    assert row["n_tot"] == 20  # median(10, 20, 30)
    assert row["n_pos"] == 4  # median(4, 6, 2)
    # ratio is the median of the per-column ratios: median(0.4, 0.3, 0.0667) = 0.3
    assert row["ratio"] == pytest.approx(0.3)
