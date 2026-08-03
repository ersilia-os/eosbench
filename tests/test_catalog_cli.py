import numpy as np
import pandas as pd
import pytest

from eosbench import get_catalog
from eosbench.dataset import catalog_columns
from eosbench.cli.catalog import (
    filter_catalog,
    _fmt_cell,
    _human_count,
    _grade_color,
    _ratio_cell,
    _skew_cell,
    _leaderboard_cell,
    _display_columns,
)


@pytest.fixture
def catalog():
    """A small synthetic catalog frame with a NaN-metric row."""
    return pd.DataFrame(
        {
            "name": ["ames", "bbbp", "herg", "sider"],
            "source": ["tdc", "moleculenet", "tdc", "moleculenet"],
            "n_tot": [7278, 2039, 655, 1427],
            "n_pos": [3974, 1560, 350, None],
            "auroc": [0.90, 0.85, 0.78, np.nan],
            "auprc": [0.91, 0.83, 0.70, np.nan],
            "ratio": [0.546, 0.765, 0.534, np.nan],
        }
    )


def test_min_max_samples_bound_rows(catalog):
    assert set(filter_catalog(catalog, min_samples=1000)["name"]) == {
        "ames",
        "bbbp",
        "sider",
    }
    assert set(filter_catalog(catalog, max_samples=1000)["name"]) == {"herg"}
    assert set(filter_catalog(catalog, min_samples=1000, max_samples=2500)["name"]) == {
        "bbbp",
        "sider",
    }


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
            "name": ["esol", "freesolv", "lipo"],
            "source": ["moleculenet", "moleculenet", "moleculenet"],
            "task": ["regression"] * 3,
            "n_tot": [1128, 642, 4200],
            "rmse": [0.55, 1.15, np.nan],
            "r2": [0.93, 0.88, np.nan],
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
        "last_updated",
    ]
    assert catalog_columns("regression") == [
        "id",
        "name",
        "source",
        "task",
        "n_columns",
        "n_tot",
        "size",
        "rmse",
        "r2",
        "skew",
        "leaderboard_score",
        "leaderboard_metric",
        "leaderboard_split",
        "leaderboard_provider",
        "last_updated",
    ]
    assert "column" in catalog_columns("regression", expand=True)
    assert "n_columns" not in catalog_columns("regression", expand=True)


def test_catalog_columns_include_leaderboard():
    assert catalog_columns("classification")[-5:] == [
        "leaderboard_score",
        "leaderboard_metric",
        "leaderboard_split",
        "leaderboard_provider",
        "last_updated",
    ]
    assert catalog_columns("regression")[-5:] == [
        "leaderboard_score",
        "leaderboard_metric",
        "leaderboard_split",
        "leaderboard_provider",
        "last_updated",
    ]


def test_get_catalog_surfaces_leaderboard_for_moleculenet():
    df = get_catalog(source="moleculenet")
    assert {
        "leaderboard_score", "leaderboard_metric", "leaderboard_split", "leaderboard_provider",
    } <= set(df.columns)
    sider = df.loc[df["name"] == "sider"].iloc[0]
    assert sider["leaderboard_metric"] == "AUROC"
    assert sider["leaderboard_score"] == pytest.approx(0.638)
    assert sider["leaderboard_split"] == "random"
    assert sider["leaderboard_provider"] == "moleculenet"


def test_get_catalog_leaderboard_for_tdcommons_from_polaris():
    df = get_catalog(source="tdcommons").set_index("name")
    # ADMET Benchmark Group tasks carry their official Polaris leaderboard score/metric.
    ames = df.loc["ames"]
    assert ames["leaderboard_metric"] == "AUROC"
    assert ames["leaderboard_score"] == pytest.approx(0.871)
    assert ames["leaderboard_split"] == "scaffold"
    assert ames["leaderboard_provider"] == "polaris"
    # clintox is cross-filled from MoleculeNet (TDC's ADMET group doesn't cover it), so its
    # leaderboard_split describes MoleculeNet's own random split, not this row's own scaffold
    # split -- leaderboard_provider is what signals that mismatch is expected.
    assert df.loc["clintox", "leaderboard_split"] == "random"
    assert df.loc["clintox", "leaderboard_provider"] == "moleculenet"
    # CYP inhibition tasks are ranked by AUPRC.
    assert df.loc["cyp2d6_veith", "leaderboard_metric"] == "AUPRC"
    # clintox is not in the ADMET group, but is cross-filled from MoleculeNet.
    assert df.loc["clintox", "leaderboard_score"] == pytest.approx(0.832)
    # Datasets with no comparable published number at all stay blank.
    assert pd.isna(df.loc["carcinogens_lagunin", "leaderboard_score"])


def test_get_catalog_unknown_source_is_empty_but_well_formed():
    df = get_catalog(source="nonexistent")
    assert df.empty
    assert "leaderboard_score" in df.columns  # headers present even with no rows


def test_get_catalog_regression_is_well_formed():
    """The regression catalog carries the regression columns and rmse/r2 (no class balance)."""
    df = get_catalog(task="regression")
    assert list(df.columns) == catalog_columns("regression")
    assert "n_pos" not in df.columns and "ratio" not in df.columns
    assert "rmse" in df.columns and "r2" in df.columns
    # MoleculeNet regression sets are now bundled (esol/freesolv/lipophilicity/qm8).
    assert not df.empty
    assert "esol" in set(df["name"])


def test_get_catalog_all_tasks_combines_both():
    """task='all' (the CLI default) unions classification + regression in one frame."""
    df = get_catalog(task="all")
    assert list(df.columns) == catalog_columns("all")
    # Both metric sets present as columns.
    assert {"auroc", "auprc", "rmse", "r2", "n_pos", "ratio"} <= set(df.columns)
    tasks = set(df["task"])
    assert {"classification", "regression"} <= tasks
    # A classification set and a regression set both appear.
    assert {"ames", "esol"} <= set(df["name"])
    # A regression row has rmse but no auroc; a classification row vice versa.
    esol = df[df["name"] == "esol"].iloc[0]
    assert pd.notna(esol["rmse"]) and pd.isna(esol["auroc"])
    ames = df[df["name"] == "ames"].iloc[0]
    assert pd.notna(ames["auroc"]) and pd.isna(ames["rmse"])


# --- cell formatting --------------------------------------------------------


def test_fmt_cell_counts_are_grouped_integers():
    assert _fmt_cell("n_tot", 10000) == "10,000"
    assert (
        _fmt_cell("n_tot", 7278.0) == "7,278"
    )  # float (from NaN-promoted col) -> no decimals
    assert _fmt_cell("n_pos", None) == "-"
    assert _fmt_cell("n_pos", np.nan) == "-"


def test_fmt_cell_metrics_keep_decimals():
    assert _fmt_cell("auroc", 0.9029) == "0.9029"
    assert _fmt_cell("ratio", np.nan) == "-"
    assert _fmt_cell("name", "ames") == "ames"


# --- richer formatting helpers ----------------------------------------------

def test_human_count_abbreviates_only_big_values():
    assert _human_count(7278) == "7,278"        # small: exact, grouped
    assert _human_count(41120) == "41,120"      # < 100k stays exact
    assert _human_count(99999) == "99,999"
    assert _human_count(100000) == "100k"       # >= 100k abbreviated
    assert _human_count(302343) == "302k"
    assert _human_count(1203045) == "1.2M"


def test_grade_color_thresholds():
    assert _grade_color(0.95) == "green"
    assert _grade_color(0.80) == "green"        # boundary inclusive
    assert _grade_color(0.70) == "yellow"
    assert _grade_color(0.60) == "yellow"
    assert _grade_color(0.55) == "red"


def test_ratio_cell_bar_length_tracks_value():
    assert _ratio_cell(0.0).count("▰") == 0 and _ratio_cell(0.0).count("▱") == 5
    assert _ratio_cell(1.0).count("▰") == 5
    assert "0.50" in _ratio_cell(0.5)
    assert _ratio_cell(None) == "[dim]-[/dim]"   # blanks dimmed


def test_skew_cell_is_center_anchored_and_fills_toward_the_skewed_side():
    assert _skew_cell(0.0) == "[dim]▱▱▰▱▱[/dim] +0.00"     # symmetric baseline: center only
    assert _skew_cell(0.49) == "[dim]▱▱▰▱▱[/dim] +0.49"    # below the fill threshold: still center-only
    assert _skew_cell(-1.17) == "[dim]▱▰▰▱▱[/dim] -1.17"   # left-tailed: fills toward the left
    assert _skew_cell(-3.0) == "[dim]▰▰▰▱▱[/dim] -3.00"    # saturates at |skew|>=2
    assert _skew_cell(3.0) == "[dim]▱▱▰▰▰[/dim] +3.00"     # right-tailed: fills toward the right
    assert _skew_cell(None) == "[dim]-[/dim]"              # blanks dimmed, same as ratio


def test_leaderboard_cell_merges_score_and_metric():
    cell = _leaderboard_cell({"leaderboard_score": 0.871, "leaderboard_metric": "AUROC"})
    assert "0.871" in cell and "AUROC" in cell
    assert "green" in cell  # 0.871 -> graded green
    assert _leaderboard_cell({"leaderboard_score": None, "leaderboard_metric": None}) == "[dim]-[/dim]"


def test_display_columns_merges_leaderboard_pair():
    df = get_catalog(source="moleculenet")
    headers = [h for h, _justify, _render in _display_columns(df)]
    assert "leaderboard" in headers
    assert "leaderboard_score" not in headers and "leaderboard_metric" not in headers
    # leaderboard_split/leaderboard_provider stay their own plain columns (display names
    # "lb_split"/"lb_provider"), right after the merged "leaderboard" one, in df order.
    assert "lb_split" in headers and "lb_provider" in headers
    assert headers.index("lb_split") == headers.index("leaderboard") + 1
    assert headers.index("lb_provider") == headers.index("lb_split") + 1
    # every renderer returns a string for a sample row (no crash)
    row = df.iloc[0]
    assert all(isinstance(render(row), str) for _h, _j, render in _display_columns(df))
