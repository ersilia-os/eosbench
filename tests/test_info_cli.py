from eosbench.cli.info import _column_detail_rows, _description_rows, _leaderboard_rows
from eosbench.dataset import DatasetInfo


def _rows_to_dict(rows):
    return dict(rows)


def test_leaderboard_rows_include_split_when_present():
    rows = _rows_to_dict(_leaderboard_rows({
        "leaderboard_value": 0.806, "leaderboard_metric": "ROC-AUC",
        "leaderboard_split": "scaffold", "leaderboard_provider": "polaris",
        "leaderboard_source": "Wu et al. 2018",
    }))
    assert rows["leaderboard"] == "0.8060 (ROC-AUC)"
    assert rows["leaderboard_split"] == "scaffold"
    assert rows["leaderboard_provider"] == "polaris"
    # absent split/provider -> no row
    bare = _rows_to_dict(_leaderboard_rows({"leaderboard_value": 0.5, "leaderboard_metric": "AUROC"}))
    assert "leaderboard_split" not in bare
    assert "leaderboard_provider" not in bare


def test_bundled_metadata_records_leaderboard_split():
    # MoleculeNet recommends scaffold for BACE, random for Tox21; TDC ADMET uses scaffold.
    assert DatasetInfo("moleculenet", "classification", "bace").metadata["leaderboard_split"] == "scaffold"
    assert DatasetInfo("moleculenet", "classification", "tox21").metadata["leaderboard_split"] == "random"
    assert DatasetInfo("tdcommons", "classification", "ames").metadata["leaderboard_split"] == "scaffold"


def test_bundled_metadata_records_leaderboard_provider():
    # Provenance: tdcommons ADMET scores come from Polaris; MoleculeNet from the paper.
    assert DatasetInfo("tdcommons", "classification", "ames").metadata["leaderboard_provider"] == "polaris"
    assert DatasetInfo("moleculenet", "classification", "bbbp").metadata["leaderboard_provider"] == "moleculenet"
    # A tdcommons dataset with no leaderboard anywhere stays blank.
    assert DatasetInfo("tdcommons", "classification", "hiv").metadata["leaderboard_provider"] is None


def test_column_detail_shows_full_description_and_computes_ratio():
    long_desc = "Adverse drug reactions in the MedDRA System Organ Class 'Hepatobiliary disorders'." * 2
    c = {
        "n_samples": 1427,
        "n_positives": 743,
        "n_negatives": 684,
        "random_auroc_mean": 0.81,
        "random_auroc_std": 0.02,
        "description": long_desc,
        "description_source": "MoleculeNet SIDER",
    }
    d = _rows_to_dict(_column_detail_rows(c))
    assert d["n_samples"] == "1,427"             # grouped integer
    assert d["ratio"] == "0.5207"                # computed n_pos / n_tot
    assert d["auroc (random split)"] == "0.8100 ± 0.0200"
    assert d["description"] == long_desc          # full text, untruncated
    assert d["description_source"] == "MoleculeNet SIDER"


def test_column_detail_surfaces_unknown_keys_verbatim():
    # Fields added to the metadata later must still appear in the --column view.
    c = {"n_samples": 10, "n_positives": 4, "assay_id": "CHEMBL1234", "units": "nM"}
    d = _rows_to_dict(_column_detail_rows(c))
    assert d["assay_id"] == "CHEMBL1234"
    assert d["units"] == "nM"


def test_column_detail_skips_missing_metrics_gracefully():
    c = {"n_samples": 5, "n_positives": 2, "n_negatives": 3}
    d = _rows_to_dict(_column_detail_rows(c))
    assert d["auroc (random split)"] == "-"
    assert "auroc (scaffold split)" not in d  # absent key -> row omitted


def test_description_rows_empty_when_absent():
    assert _description_rows({"n_samples": 5}) == []
