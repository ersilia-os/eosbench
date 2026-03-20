import numpy as np
import pandas as pd
import pytest

from eosbench import (
    get_catalog,
    load_dataset,
    mirror_dataset,
)


def test_get_catalog_returns_summary_dataframe():
    catalog = get_catalog()

    assert list(catalog.columns) == [
        "name",
        "source",
        "task",
        "n_tot",
        "n_pos",
        "auroc",
        "aupr",
        "ratio",
    ]

    ames = catalog.loc[catalog["name"] == "ames"].iloc[0]
    assert ames["source"] == "tdc"
    assert ames["n_tot"] == 7278
    assert ames["n_pos"] == 3974
    assert ames["auroc"] == pytest.approx(0.9029)
    assert ames["aupr"] == pytest.approx(0.9132)
    assert ames["ratio"] == pytest.approx(3974 / 7278)


def test_mirror_dataset_materializes_expected_folder_structure(tmp_path, monkeypatch):
    source = "tdc"
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

    created = mirror_dataset(source, dataset, featurization="morgan", output_dir=tmp_path / "data", task_type=task_type)

    assert created == tmp_path / "data" / source / task_type / dataset
    assert (created / "data.csv").is_file()
    assert (created / "folds.csv").is_file()
    assert (created / "morgan.npy").is_file()
    assert (created / "metadata.json").is_file()


def test_mirror_dataset_rejects_unknown_featurization(tmp_path):
    with pytest.raises(ValueError, match="featurization must be None"):
        mirror_dataset("tdc", "ames", featurization="bogus", output_dir=tmp_path / "data")


def test_load_dataset_supports_smiles_and_feature_arrays(tmp_path, monkeypatch):
    source = "tdc"
    task_type = "classification"
    dataset = "ames"

    csv_path = tmp_path / "data.csv"
    npy_path = tmp_path / "morgan.npy"
    chemeleon_path = tmp_path / "chemeleon.npy"
    folds_path = tmp_path / "folds.csv"

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "activity": [1.5, 0.0, 1.0],
        }
    ).to_csv(csv_path, index=False)
    np.save(npy_path, np.arange(12).reshape(3, 4))
    np.save(chemeleon_path, np.arange(12, 24).reshape(3, 4))
    pd.DataFrame({"fold": [0, 1, 0]}).to_csv(folds_path, index=False)

    def fake_fetch(src, task, name, filename):
        assert (src, task, name) == (source, task_type, dataset)
        mapping = {
            "data.csv": csv_path,
            "morgan.npy": npy_path,
            "chemeleon.npy": chemeleon_path,
            "folds.csv": folds_path,
        }
        return str(mapping[filename])

    monkeypatch.setattr("eosbench.dataset._fetch", fake_fetch)

    ds_smiles = load_dataset(source, dataset, featurization=None, task_type=task_type)
    ds_morgan = load_dataset(source, dataset, featurization="morgan", task_type=task_type)
    ds_chemeleon = load_dataset(source, dataset, featurization="chemeleon", task_type=task_type)

    assert ds_smiles.X == ["CCO", "CCN", "CCC"]
    assert ds_smiles.y.tolist() == [1.5, 0.0, 1.0]
    assert len(ds_smiles.split) > 0
    assert isinstance(ds_smiles.metadata, dict)

    assert ds_morgan.X.shape == (3, 4)
    assert ds_chemeleon.X.shape == (3, 4)


def test_load_dataset_rejects_unknown_featurization():
    with pytest.raises(ValueError, match="featurization must be None"):
        load_dataset("tdc", "ames", featurization="bogus")
