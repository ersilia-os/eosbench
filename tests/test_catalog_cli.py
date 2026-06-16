import numpy as np
import pandas as pd
import pytest

from eosbench import get_catalog
from eosbench.dataset import catalog_columns
from eosbench.cli.catalog import filter_catalog, _fmt_cell


@pytest.fixture
def catalog():
    """A small synthetic catalog frame with a NaN-metric row."""
    return pd.DataFrame(
        {
            "name":   ["ames", "bbbp", "herg", "sider"],
            "source": ["tdc", "moleculenet", "tdc", "moleculenet"],
            "n_tot":  [7278, 2039, 655, 1427],
            "n_pos":  [3974, 1560, 350, None],
            "auroc":  [0.90, 0.85, 0.78, np.nan],
            "auprc":  [0.91, 0.83, 0.70, np.nan],
            "ratio":  [0.546, 0.765, 0.534, np.nan],
        }
    )


def test_min_max_samples_bound_rows(catalog):
    assert set(filter_catalog(catalog, min_samples=1000)["name"]) == {"ames", "bbbp", "sider"}
    assert set(filter_catalog(catalog, max_samples=1000)["name"]) == {"herg"}
    assert set(filter_catalog(catalog, min_samples=1000, max_samples=2500)["name"]) == {"bbbp", "sider"}


def test_min_max_ratio_bound_rows(catalog):
    assert set(filter_catalog(catalog, min_ratio=0.6)["name"]) == {"bbbp"}
    assert set(filter_catalog(catalog, max_ratio=0.55)["name"]) == {"ames", "herg"}


def test_min_auroc_drops_below_threshold_and_nan(catalog):
    kept = set(filter_catalog(catalog, min_auroc=0.8)["name"])
    assert kept == {"ames", "bbbp"}  # herg=0.78 excluded, sider=NaN dropped


def test_max_auroc_and_max_auprc_drop_above_threshold_and_nan(catalog):
    assert set(filter_catalog(catalog, max_auroc=0.8)["name"]) == {"herg"}
    assert set(filter_catalog(catalog, max_auprc=0.85)["name"]) == {"bbbp", "herg"}


def test_min_auprc_drops_below_threshold_and_nan(catalog):
    assert set(filter_catalog(catalog, min_auprc=0.85)["name"]) == {"ames"}


def test_name_substring_is_case_insensitive(catalog):
    assert list(filter_catalog(catalog, name="AME")["name"]) == ["ames"]
    assert filter_catalog(catalog, name="zzz").empty


def test_sort_by_and_desc_and_limit(catalog):
    ordered = filter_catalog(catalog, sort_by="n_tot", desc=True)["name"].tolist()
    assert ordered == ["ames", "bbbp", "sider", "herg"]

    ordered_asc = filter_catalog(catalog, sort_by="auroc")["name"].tolist()
    # NaN sorts last regardless of direction (pandas default).
    assert ordered_asc[:3] == ["herg", "bbbp", "ames"]
    assert ordered_asc[-1] == "sider"

    top2 = filter_catalog(catalog, sort_by="n_tot", desc=True, limit=2)["name"].tolist()
    assert top2 == ["ames", "bbbp"]


def test_filters_compose_with_and(catalog):
    kept = filter_catalog(catalog, min_samples=1000, max_ratio=0.6)["name"].tolist()
    assert kept == ["ames"]  # bbbp ratio too high, sider ratio NaN, herg too small


def test_sort_by_missing_column_raises(catalog):
    with pytest.raises(ValueError, match="cannot sort by"):
        filter_catalog(catalog, sort_by="column")  # expand-only; absent here


def test_empty_result_is_not_an_error(catalog):
    out = filter_catalog(catalog, min_samples=10**9)
    assert out.empty
    assert list(out.columns) == list(catalog.columns)


def test_filter_catalog_works_on_real_get_catalog():
    """End-to-end against packaged metadata (no network)."""
    df = get_catalog(source="tdcommons")
    out = filter_catalog(df, name="ames", sort_by="n_tot", desc=True, limit=1)
    assert list(out["name"]) == ["ames"]


# --- regression / task-aware columns ---------------------------------------

@pytest.fixture
def regression_catalog():
    """A synthetic regression catalog: rmse/r2 metrics, no n_pos/ratio."""
    return pd.DataFrame(
        {
            "name":   ["esol", "freesolv", "lipo"],
            "source": ["moleculenet", "moleculenet", "moleculenet"],
            "task":   ["regression"] * 3,
            "n_tot":  [1128, 642, 4200],
            "rmse":   [0.55, 1.15, np.nan],
            "r2":     [0.93, 0.88, np.nan],
        }
    )


def test_regression_metric_filters(regression_catalog):
    assert set(filter_catalog(regression_catalog, max_rmse=1.0)["name"]) == {"esol"}
    assert set(filter_catalog(regression_catalog, min_rmse=1.0)["name"]) == {"freesolv"}
    # NaN-metric rows dropped by threshold filters.
    assert set(filter_catalog(regression_catalog, min_r2=0.9)["name"]) == {"esol"}


def test_classification_filter_on_regression_frame_errors(regression_catalog):
    with pytest.raises(ValueError, match="Check --task"):
        filter_catalog(regression_catalog, min_auroc=0.8)
    with pytest.raises(ValueError, match="Check --task"):
        filter_catalog(regression_catalog, min_ratio=0.5)


def test_catalog_columns_are_task_aware():
    assert catalog_columns("classification") == [
        "name", "source", "task", "n_columns", "n_tot", "n_pos",
        "auroc", "auprc", "ratio", "leaderboard_score", "leaderboard_metric", "last_updated",
    ]
    assert catalog_columns("regression") == [
        "name", "source", "task", "n_columns", "n_tot", "rmse", "r2",
        "leaderboard_score", "leaderboard_metric", "last_updated",
    ]
    assert "column" in catalog_columns("regression", expand=True)
    assert "n_columns" not in catalog_columns("regression", expand=True)


def test_catalog_columns_include_leaderboard():
    assert catalog_columns("classification")[-3:] == [
        "leaderboard_score", "leaderboard_metric", "last_updated",
    ]
    assert catalog_columns("regression")[-3:] == [
        "leaderboard_score", "leaderboard_metric", "last_updated",
    ]


def test_get_catalog_surfaces_leaderboard_for_moleculenet():
    df = get_catalog(source="moleculenet")
    assert {"leaderboard_score", "leaderboard_metric"} <= set(df.columns)
    sider = df.loc[df["name"] == "sider"].iloc[0]
    assert sider["leaderboard_metric"] == "ROC-AUC"
    assert sider["leaderboard_score"] == pytest.approx(0.638)


def test_get_catalog_leaderboard_for_tdcommons_from_polaris():
    df = get_catalog(source="tdcommons").set_index("name")
    # ADMET Benchmark Group tasks carry their official Polaris leaderboard score/metric.
    ames = df.loc["ames"]
    assert ames["leaderboard_metric"] == "AUROC"
    assert ames["leaderboard_score"] == pytest.approx(0.871)
    # CYP inhibition tasks are ranked by AUPRC.
    assert df.loc["cyp2d6_veith", "leaderboard_metric"] == "AUPRC"
    # Datasets outside the ADMET Benchmark Group stay blank.
    assert pd.isna(df.loc["clintox", "leaderboard_score"])
    assert pd.isna(df.loc["cyp1a2_veith", "leaderboard_score"])


def test_get_catalog_unknown_source_is_empty_but_well_formed():
    df = get_catalog(source="nonexistent")
    assert df.empty
    assert "leaderboard_score" in df.columns  # headers present even with no rows


def test_get_catalog_regression_is_empty_but_well_formed():
    """No regression data is bundled yet; must not crash and must carry headers."""
    df = get_catalog(task="regression")
    assert df.empty
    assert list(df.columns) == catalog_columns("regression")
    assert "n_pos" not in df.columns and "ratio" not in df.columns


# --- cell formatting --------------------------------------------------------

def test_fmt_cell_counts_are_grouped_integers():
    assert _fmt_cell("n_tot", 10000) == "10,000"
    assert _fmt_cell("n_tot", 7278.0) == "7,278"     # float (from NaN-promoted col) -> no decimals
    assert _fmt_cell("n_pos", None) == "-"
    assert _fmt_cell("n_pos", np.nan) == "-"


def test_fmt_cell_metrics_keep_decimals():
    assert _fmt_cell("auroc", 0.9029) == "0.9029"
    assert _fmt_cell("ratio", np.nan) == "-"
    assert _fmt_cell("name", "ames") == "ames"
