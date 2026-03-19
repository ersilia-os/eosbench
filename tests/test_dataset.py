import numpy as np
import pandas as pd
import pytest

from eosbench import (
    available_datasets,
    dataset_catalog,
    fetch_datasets,
    list_datasets,
    load_dataset,
)
from eosbench.dataset import DatasetInfo, iter_datasets


def test_list_datasets_exposes_packaged_metadata():
    datasets = list_datasets()

    assert {"source": "tdc", "dataset": "ames", "task_type": "classification"} in datasets
    assert {"source": "chembl", "dataset": "chembl4649948", "task_type": "classification"} in datasets


def test_available_datasets_returns_full_sorted_name_list():
    datasets = available_datasets()

    assert "ames" in datasets
    assert "chembl4649948" in datasets
    assert datasets == sorted(datasets)

    tdc_datasets = available_datasets(source="tdc")
    assert "ames" in tdc_datasets
    assert "chembl4649948" not in tdc_datasets


def test_dataset_catalog_returns_summary_dataframe():
    catalog = dataset_catalog()

    assert list(catalog.columns) == [
        "name",
        "source",
        "samples",
        "auroc",
        "aupr",
        "n_pos",
        "n_tot",
        "ratio",
    ]

    ames = catalog.loc[catalog["name"] == "ames"].iloc[0]
    assert ames["source"] == "tdc"
    assert ames["samples"] == 7278
    assert ames["auroc"] == pytest.approx(0.9029)
    assert ames["aupr"] == pytest.approx(0.9132)
    assert ames["n_pos"] == 3974
    assert ames["n_tot"] == 7278
    assert ames["ratio"] == pytest.approx(3974 / 7278)


def test_fetch_datasets_materializes_expected_folder_structure(tmp_path, monkeypatch):
    source = "tdc"
    task_type = "classification"
    dataset = "ames"
    remote_root = tmp_path / "remote"

    remote_root.mkdir()
    (remote_root / "data.csv").write_text("smiles,activity\nCCO,1\n")
    np.save(remote_root / "morgan.npy", np.arange(4).reshape(1, 4))
    np.save(remote_root / "chemeleon.npy", np.arange(4, 8).reshape(1, 4))

    def fake_download(src, task, name, filename, dest):
        assert (src, task, name) == (source, task_type, dataset)
        src_path = remote_root / filename
        pdest = type(src_path)(dest)
        pdest.write_bytes(src_path.read_bytes())
        return str(pdest)

    monkeypatch.setattr("eosbench.dataset._download_to", fake_download)

    created = fetch_datasets([dataset], output_dir=tmp_path / "data")

    assert created == [tmp_path / "data" / source / task_type / dataset]
    assert (created[0] / "data.csv").is_file()
    assert (created[0] / "morgan.npy").is_file()
    assert (created[0] / "chemeleon.npy").is_file()
    assert (created[0] / "metadata.json").is_file()
    assert (created[0] / "folds.csv").is_file()


def test_fetch_datasets_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_datasets(["not_a_dataset"], output_dir=tmp_path / "data")


def test_iter_datasets_yields_dataset_info():
    infos = list(iter_datasets("tdc"))

    assert infos
    assert isinstance(infos[0], DatasetInfo)
    assert infos[0].source == "tdc"
    assert infos[0].task_type == "classification"
    assert "n_samples" in infos[0].metadata


def test_load_dataset_supports_smiles_and_feature_arrays(tmp_path, monkeypatch):
    source = "tdc"
    task_type = "classification"
    dataset = "ames"

    csv_path = tmp_path / "data.csv"
    npy_path = tmp_path / "morgan.npy"
    chemeleon_path = tmp_path / "chemeleon.npy"

    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "activity": [1.5, 0.0, 1.0],
        }
    ).to_csv(csv_path, index=False)
    np.save(npy_path, np.arange(12).reshape(3, 4))
    np.save(chemeleon_path, np.arange(12, 24).reshape(3, 4))

    def fake_fetch(src, task, name, filename):
        assert (src, task, name) == (source, task_type, dataset)
        mapping = {
            "data.csv": csv_path,
            "morgan.npy": npy_path,
            "chemeleon.npy": chemeleon_path,
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


def test_dataset_info_load_delegates_to_load_dataset(monkeypatch):
    info = DatasetInfo("tdc", "classification", "ames")
    sentinel = object()

    def fake_load_dataset(source, dataset, featurization, task_type):
        assert source == "tdc"
        assert dataset == "ames"
        assert featurization == "morgan"
        assert task_type == "classification"
        return sentinel

    monkeypatch.setattr("eosbench.dataset.load_dataset", fake_load_dataset)

    assert info.load() is sentinel
