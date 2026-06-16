import argparse
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..dataset import get_catalog

# Columns that may appear in a get_catalog() frame (collapsed or expand=True,
# classification or regression). --sort_by is validated against the actual
# frame at runtime, so listing the superset here is fine.
SORT_COLUMNS = [
    "name",
    "source",
    "task",
    "column",
    "n_columns",
    "n_tot",
    "n_pos",
    "auroc",
    "auprc",
    "rmse",
    "r2",
    "ratio",
    "leaderboard_score",
    "leaderboard_metric",
    "last_updated",
]

# Integer count columns: render with thousands separators and no decimals.
COUNT_COLUMNS = {"n_tot", "n_pos", "n_columns"}
# Columns rendered right-aligned (numeric).
NUMERIC_COLUMNS = COUNT_COLUMNS | {
    "auroc",
    "auprc",
    "rmse",
    "r2",
    "ratio",
    "leaderboard_score",
}
# Friendlier header labels for the rendered table. The underlying DataFrame
# column names (used by --sort_by and the library API) are unchanged.
DISPLAY_NAMES = {
    "name": "dataset",
    "n_columns": "columns",
    "leaderboard_score": "lb_score",
    "leaderboard_metric": "lb_metric",
}


def _threshold(df, col, value, op, flag):
    """Apply a >=/<= numeric filter on ``col``; NaN/None rows are dropped.

    Raises if ``col`` is absent from the frame (e.g. a classification-only
    metric filter applied to a regression catalog), with a task-aware hint.
    """
    if value is None:
        return df
    if col not in df.columns:
        raise ValueError(
            f"{flag} cannot be applied: {col!r} is not a column in this catalog "
            f"(columns: {', '.join(df.columns)}). Check --task."
        )
    series = pd.to_numeric(df[col], errors="coerce")
    return df[series >= value] if op == ">=" else df[series <= value]


def filter_catalog(
    df,
    *,
    name=None,
    min_samples=None,
    max_samples=None,
    min_ratio=None,
    max_ratio=None,
    min_auroc=None,
    max_auroc=None,
    min_auprc=None,
    max_auprc=None,
    min_rmse=None,
    max_rmse=None,
    min_r2=None,
    max_r2=None,
    sort_by=None,
    desc=False,
    limit=None,
):
    """Filter, sort, and limit a ``get_catalog()`` DataFrame.

    Pure function — no I/O. Filters compose with logical AND. Threshold filters
    on metric/ratio columns drop rows with missing (NaN/None) values, since a
    dataset with an unknown value cannot be shown to clear a threshold. A metric
    filter whose column is absent for the current task raises a clear error.
    """
    if name is not None:
        df = df[df["name"].str.contains(name, case=False, na=False)]
    df = _threshold(df, "n_tot", min_samples, ">=", "--min_samples")
    df = _threshold(df, "n_tot", max_samples, "<=", "--max_samples")
    df = _threshold(df, "ratio", min_ratio, ">=", "--min_ratio")
    df = _threshold(df, "ratio", max_ratio, "<=", "--max_ratio")
    df = _threshold(df, "auroc", min_auroc, ">=", "--min_auroc")
    df = _threshold(df, "auroc", max_auroc, "<=", "--max_auroc")
    df = _threshold(df, "auprc", min_auprc, ">=", "--min_auprc")
    df = _threshold(df, "auprc", max_auprc, "<=", "--max_auprc")
    df = _threshold(df, "rmse", min_rmse, ">=", "--min_rmse")
    df = _threshold(df, "rmse", max_rmse, "<=", "--max_rmse")
    df = _threshold(df, "r2", min_r2, ">=", "--min_r2")
    df = _threshold(df, "r2", max_r2, "<=", "--max_r2")

    if sort_by is not None:
        if sort_by not in df.columns:
            raise ValueError(
                f"cannot sort by {sort_by!r}: column not present "
                f"(available: {list(df.columns)})"
            )
        df = df.sort_values(sort_by, ascending=not desc)

    if limit is not None:
        df = df.head(limit)

    return df


def _is_missing(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def _fmt_cell(col, v):
    """Format a single table cell: counts as grouped integers, metrics as floats."""
    if _is_missing(v):
        return "-"
    if col in COUNT_COLUMNS:
        return f"{int(round(float(v))):,}"  # 10000 -> "10,000"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


_EPILOG = """\
columns (which appear depends on --task; --sort_by key in parentheses):
  dataset       Dataset (family) identifier, e.g. "ames", "sider". (key: name)
  source        Where the data comes from: tdcommons, moleculenet or polaris.
  task          Task type: classification or regression.
  columns       Number of label columns (endpoints) in the family; 1 for
                single-label sets. (key: n_columns) [default view only]
  column        Name of the individual label column / endpoint. [--expand only]
  n_tot         Total samples; for multi-column families, the molecule count.
  n_pos         Positive-class samples.                [classification only]
  ratio         Positive-class fraction, n_pos / n_tot. [classification only]
  auroc, auprc  Baseline AUROC / AUPRC.                [classification]
  rmse, r2      Baseline RMSE / R-squared.             [regression]
  lb_score      Best published leaderboard score, where known. (key: leaderboard_score)
  lb_metric     Metric that lb_score is measured in, e.g. ROC-AUC, PRC-AUC.
                (key: leaderboard_metric)
  last_updated  Date the dataset entry was last refreshed (YYYY-MM-DD).

The auroc/auprc/rmse/r2 columns are a RandomForest baseline averaged over random
K-fold cross-validation — a reference floor, not the best published model. The
lb_* columns are that best published model, where known (MoleculeNet, and TDC
ADMET tasks via Polaris); blank otherwise.

examples:
  eosbench catalog                                  list every dataset
  eosbench catalog --task regression --sort_by rmse  regression sets, lowest RMSE first
  eosbench catalog --name cyp                        datasets whose name contains "cyp"
  eosbench catalog --min_samples 1000 --max_ratio 0.5
                                                    big, imbalanced datasets
  eosbench catalog --min_auroc 0.8 --sort_by auroc --desc
                                                    strong baselines, best first
  eosbench catalog --sort_by n_tot --desc --limit 5  the 5 largest datasets
  eosbench catalog --expand                          one row per label column (multi-column sets)

Threshold filters drop datasets with a missing value for that column (e.g.
--min_auroc skips datasets with no recorded AUROC). Filters combine with AND.
"""


def main():
    parser = argparse.ArgumentParser(
        prog="eosbench catalog",
        description="List available eosbench datasets as a table, with optional "
        "filtering, sorting and limiting.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    selection = parser.add_argument_group("dataset selection")
    selection.add_argument(
        "--source",
        type=str,
        default=None,
        metavar="SOURCE",
        help="Only this source, e.g. tdcommons, moleculenet, polaris (default: all).",
    )
    selection.add_argument(
        "--task",
        type=str,
        default="classification",
        metavar="TASK",
        help="Task type: classification or regression (default: classification).",
    )
    selection.add_argument(
        "--expand",
        action="store_true",
        help="One row per label column instead of one row per dataset family.",
    )

    filters = parser.add_argument_group("filters (combine with AND)")
    filters.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="SUBSTR",
        help="Keep datasets whose name contains SUBSTR (case-insensitive).",
    )
    filters.add_argument(
        "--min_samples",
        type=int,
        default=None,
        metavar="N",
        help="Keep datasets with n_tot >= N.",
    )
    filters.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Keep datasets with n_tot <= N.",
    )
    filters.add_argument(
        "--min_ratio",
        type=float,
        default=None,
        metavar="R",
        help="[classification] Keep datasets with positive-class ratio >= R.",
    )
    filters.add_argument(
        "--max_ratio",
        type=float,
        default=None,
        metavar="R",
        help="[classification] Keep datasets with positive-class ratio <= R.",
    )
    filters.add_argument(
        "--min_auroc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUROC >= A.",
    )
    filters.add_argument(
        "--max_auroc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUROC <= A.",
    )
    filters.add_argument(
        "--min_auprc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUPRC >= A.",
    )
    filters.add_argument(
        "--max_auprc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUPRC <= A.",
    )
    filters.add_argument(
        "--min_rmse",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline RMSE >= V.",
    )
    filters.add_argument(
        "--max_rmse",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline RMSE <= V.",
    )
    filters.add_argument(
        "--min_r2",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline R-squared >= V.",
    )
    filters.add_argument(
        "--max_r2",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline R-squared <= V.",
    )

    ordering = parser.add_argument_group("sorting and limiting")
    ordering.add_argument(
        "--sort_by",
        type=str,
        default=None,
        choices=SORT_COLUMNS,
        metavar="COLUMN",
        help="Sort by a column (default: source then name). Choices: "
        + ", ".join(SORT_COLUMNS)
        + ".",
    )
    ordering.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order (default: ascending).",
    )
    ordering.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Show only the first N rows (after filtering and sorting).",
    )

    args = parser.parse_args()

    df = get_catalog(source=args.source, task=args.task, expand=args.expand)
    try:
        df = filter_catalog(
            df,
            name=args.name,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            min_auroc=args.min_auroc,
            max_auroc=args.max_auroc,
            min_auprc=args.min_auprc,
            max_auprc=args.max_auprc,
            min_rmse=args.min_rmse,
            max_rmse=args.max_rmse,
            min_r2=args.min_r2,
            max_r2=args.max_r2,
            sort_by=args.sort_by,
            desc=args.desc,
            limit=args.limit,
        )
    except ValueError as e:
        parser.error(str(e))

    table = Table(title="eosbench datasets")
    for col in df.columns:
        table.add_column(
            DISPLAY_NAMES.get(col, col),
            style="cyan" if col == "name" else None,
            justify="right" if col in NUMERIC_COLUMNS else "left",
        )

    for _, row in df.iterrows():
        table.add_row(*[_fmt_cell(col, row[col]) for col in df.columns])

    Console().print(table)

    if df.empty:
        print("No datasets matched the given filters.", file=sys.stderr)


if __name__ == "__main__":
    main()
