"""Tests for the dataset-preparation splitters (scripts/_prepare_common.py).

These require the optional ``prepare`` extras (rdkit, scikit-learn); the module is
skipped when they are not installed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import _prepare_common as common  # noqa: E402

# Distinct-scaffold molecules: many positives, few negatives — the skewed shape that
# made a naive size-descending scaffold split produce a single-class test set (BBBP).
_POS_SMILES = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "c1ccoc1",
    "c1cc2ccccc2c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOCC1",
    "c1ccc2ccccc2c1",
    "c1ccc2ncccc2c1",
]
_NEG_SMILES = ["C1CCCC1", "C1CCNC1", "C1CCOC1"]


def test_scaffold_split_keeps_both_classes_in_test():
    smiles = _POS_SMILES + _NEG_SMILES
    y = np.array([1] * len(_POS_SMILES) + [0] * len(_NEG_SMILES))
    # Mirror the pipeline: parse first, then split the surviving molecules.
    _, mols, keep = common.parse_molecules(smiles)
    y = y[keep]
    labels = common.scaffold_split(mols, y)

    test = labels == "test"
    assert test.sum() > 0
    # The whole point of the stratified split: test is never single-class.
    assert set(y[test].tolist()) == {0, 1}
    assert set(y[~test].tolist()) == {0, 1}


def test_make_random_folds_stratified_keeps_both_classes_per_fold():
    y = np.array([1] * 40 + [0] * 10)
    folds = common.make_random_folds(len(y), k=5, seed=42, stratify=y)

    assert len(folds) == len(y)
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    for k in range(5):  # stratified: every fold carries both classes
        assert set(y[folds == k].tolist()) == {0, 1}


def test_make_random_folds_plain_covers_all_samples():
    folds = common.make_random_folds(50, k=5, seed=42)  # no stratify (multi-task case)
    assert len(folds) == 50
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]


def test_make_random_folds_is_deterministic():
    assert (
        common.make_random_folds(50, k=5, seed=42, stratify=None).tolist()
        == common.make_random_folds(50, k=5, seed=42, stratify=None).tolist()
    )


def test_prepare_family_shares_split_across_tasks(tmp_path, monkeypatch):
    """A multi-task family writes one shared split and one feature matrix for all tasks."""
    monkeypatch.setattr(common, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(common, "PKG_DATA_ROOT", tmp_path / "pkg")

    smiles = _POS_SMILES + _NEG_SMILES
    n = len(smiles)
    # Two tasks over the SAME molecules; task B leaves some rows unlabeled (NaN).
    task_a = [1] * len(_POS_SMILES) + [0] * len(_NEG_SMILES)
    task_b = [1, 0] * (n // 2) + [1] * (n % 2)
    label_df = pd.DataFrame({"A": task_a, "B": [float(v) for v in task_b]})
    label_df.loc[:2, "B"] = np.nan  # vary missingness between tasks

    out = common.prepare_family(
        source="moleculenet",
        family="demo",
        smiles=smiles,
        label_df=label_df,
        n_folds=3,
        seed=42,
    )

    data = pd.read_csv(out / "data.csv")
    folds = pd.read_csv(out / "folds.csv")
    with open(out / "metadata.json") as f:
        meta = json.load(f)

    assert list(data.columns) == ["smiles", "A", "B"]
    assert set(folds.columns) == {"random_fold", "scaffold_split"}
    assert len(folds) == len(data)  # one conserved split per molecule
    assert meta["n_columns"] == 2
    assert set(meta["columns"]) == {"A", "B"}
    assert meta["split"]["conserved"] is True
    assert meta["split"]["stratified"] is False  # multi-task -> not stratified
    assert data["B"].isna().any()  # NaN preserved for the unlabeled rows
    assert (
        common.PKG_DATA_ROOT
        / "moleculenet"
        / "classification"
        / "demo"
        / "metadata.json"
    ).exists()


def test_column_descriptions_for_curated_and_templated_sets():
    import moleculenet_descriptions as desc  # noqa: E402

    tox21 = desc.describe_columns("tox21", ["NR-AR", "SR-p53"])
    assert "androgen receptor" in tox21["NR-AR"]["description"].lower()
    assert tox21["NR-AR"]["description_source"].startswith("Tox21")

    sider = desc.describe_columns("sider", ["Hepatobiliary disorders"])
    assert "Hepatobiliary disorders" in sider["Hepatobiliary disorders"]["description"]
    assert "MedDRA" in sider["Hepatobiliary disorders"]["description_source"]

    muv = desc.describe_columns("muv", ["MUV-466"])
    assert "S1P1" in muv["MUV-466"]["description"]


def test_toxcast_legacy_endpoints_are_all_described():
    """The 21 MoleculeNet-era ToxCast names absent from invitroDB v3.3 have curated text."""
    import moleculenet_descriptions as desc  # noqa: E402

    legacy = [
        "ACEA_T47D_80hr_Negative",
        "ACEA_T47D_80hr_Positive",
        "APR_Hepat_Apoptosis_24hr_up",
        "APR_Hepat_Apoptosis_48hr_up",
        "APR_Hepat_CellLoss_24hr_dn",
        "APR_Hepat_CellLoss_48hr_dn",
        "APR_Hepat_DNADamage_24hr_up",
        "APR_Hepat_DNADamage_48hr_up",
        "APR_Hepat_DNATexture_24hr_up",
        "APR_Hepat_DNATexture_48hr_up",
        "APR_Hepat_MitoFxnI_1hr_dn",
        "APR_Hepat_MitoFxnI_24hr_dn",
        "APR_Hepat_MitoFxnI_48hr_dn",
        "APR_Hepat_NuclearSize_24hr_dn",
        "APR_Hepat_NuclearSize_48hr_dn",
        "APR_Hepat_Steatosis_24hr_up",
        "APR_Hepat_Steatosis_48hr_up",
        "TOX21_AR_LUC_MDAKB2_Antagonist",
        "TOX21_AR_LUC_MDAKB2_Antagonist2",
        "TOX21_ERa_LUC_BG1_Agonist",
        "TOX21_ERa_LUC_BG1_Antagonist",
    ]
    assert set(legacy) <= set(desc.TOXCAST_LEGACY)
    assert all(desc.TOXCAST_LEGACY[c].strip() for c in legacy)


def test_prepare_family_uses_injected_holdout(tmp_path, monkeypatch):
    """A source-provided holdout (e.g. Polaris) is used verbatim instead of a scaffold split."""
    monkeypatch.setattr(common, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(common, "PKG_DATA_ROOT", tmp_path / "pkg")

    smiles = _POS_SMILES + _NEG_SMILES
    y = [1] * len(_POS_SMILES) + [0] * len(_NEG_SMILES)
    label_df = pd.DataFrame({"act": y})
    # Explicit official split aligned to `smiles` (last three molecules are the test set).
    holdout = np.array(["train"] * (len(smiles) - 3) + ["test"] * 3, dtype=object)

    out = common.prepare_family(
        source="polaris",
        family="demo",
        smiles=smiles,
        label_df=label_df,
        n_folds=3,
        seed=42,
        holdout=holdout,
        holdout_method="polaris",
    )

    folds = pd.read_csv(out / "folds.csv")
    with open(out / "metadata.json") as f:
        meta = json.load(f)

    # The holdout passes through verbatim, filtered to the molecules that parse.
    _, _, keep = common.parse_molecules(smiles)
    assert folds["scaffold_split"].tolist() == holdout[keep].tolist()
    assert meta["split"]["scaffold_split_method"] == "polaris"
    assert meta["columns"]["act"]["scaffold_split_method"] == "polaris"
    # Random K-fold is still generated alongside the official holdout.
    assert sorted(set(folds["random_fold"].tolist())) == [0, 1, 2]
