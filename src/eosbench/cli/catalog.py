import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..dataset import get_catalog, check_availability

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
    "size",
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
    "size",
    "leaderboard_score",
}
# Friendlier header labels for the rendered table. The underlying DataFrame
# column names (used by --sort_by and the library API) are unchanged.
DISPLAY_NAMES = {
    "name": "dataset",
    "n_columns": "columns",
    "leaderboard_score": "lb_score",
    "leaderboard_metric": "lb_metric",
    "available": "on S3",
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


def _human_count(n) -> str:
    """Grouped integer below 100k; abbreviated (302k / 1.2M) at/above, to tame the
    giant HTS sets without losing precision on the small/medium datasets."""
    n = int(round(float(n)))
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 100_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:,}"


def _fmt_cell(col, v):
    """Format a single table cell: counts as grouped integers, metrics as floats."""
    if _is_missing(v):
        return "-"
    if col in COUNT_COLUMNS:
        return _human_count(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# --- richer per-column display rendering ------------------------------------
# Source -> Rich color; unknown sources fall back to a small rotating palette.
_SOURCE_COLORS = {"tdcommons": "green", "moleculenet": "blue", "polaris": "magenta"}
_FALLBACK_COLORS = ["cyan", "yellow", "red", "white"]


def _source_color(src: str) -> str:
    if src in _SOURCE_COLORS:
        return _SOURCE_COLORS[src]
    return _FALLBACK_COLORS[hash(str(src)) % len(_FALLBACK_COLORS)]


def _grade_color(v: float) -> str:
    """Green/yellow/red for a [0,1] performance metric (higher is better)."""
    if v >= 0.8:
        return "green"
    if v >= 0.6:
        return "yellow"
    return "red"


def _ratio_cell(v) -> str:
    """A 5-segment unicode bar + 2-dp value as a class-balance cue."""
    if _is_missing(v):
        return "[dim]-[/dim]"
    filled = max(0, min(5, round(float(v) * 5)))
    bar = "▰" * filled + "▱" * (5 - filled)
    return f"[dim]{bar}[/dim] {float(v):.2f}"


def _skew_cell(v) -> str:
    """Center-anchored bar + signed value: the regression analog of the class-balance
    cue. Fills left for negative (left-tailed) skew, right for positive; |skew|>=2 saturates."""
    if _is_missing(v):
        return "[dim]-[/dim]"
    s = float(v)
    fill = round(min(abs(s) / 2.0, 1.0) * 2)  # 0, 1, or 2 cells
    if s < 0:
        left, right = "▱" * (2 - fill) + "▰" * fill, "▱▱"
    else:
        left, right = "▱▱", "▰" * fill + "▱" * (2 - fill)
    return f"[dim]{left}▏{right}[/dim] {s:+.2f}"


def _size_cell(v) -> str:
    """Human-readable on-disk size (full dataset incl. fingerprint matrices)."""
    if _is_missing(v):
        return "[dim]-[/dim]"
    n = float(v)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def _balance_cell(row) -> str:
    """One label-distribution column, task-aware: class-balance ratio for classification
    rows, target skewness for regression rows (they never co-occur on a row)."""
    if row.get("task") == "regression":
        return _skew_cell(row.get("skew"))
    return _ratio_cell(row.get("ratio"))


def _metric_cell(col, v) -> str:
    """Color-graded metric (auroc/auprc/r2/leaderboard score). rmse left ungraded."""
    if _is_missing(v):
        return "[dim]-[/dim]"
    text = f"{float(v):.3f}"
    if col == "rmse":
        return text
    return f"[{_grade_color(float(v))}]{text}[/]"


def _leaderboard_cell(row) -> str:
    """Merge leaderboard_score + leaderboard_metric into one cell: '0.871 AUROC'."""
    score, metric = row.get("leaderboard_score"), row.get("leaderboard_metric")
    if _is_missing(score):
        return "[dim]-[/dim]"
    cell = f"[{_grade_color(float(score))}]{float(score):.3f}[/]"
    if not _is_missing(metric):
        cell += f" [dim]{escape(str(metric))}[/dim]"
    return cell


def _source_cell(_col, v) -> str:
    return "[dim]-[/dim]" if _is_missing(v) else f"[{_source_color(v)}]{escape(str(v))}[/]"


def _plain_cell(col, v) -> str:
    return "[dim]-[/dim]" if _is_missing(v) else escape(_fmt_cell(col, v))


# The task-specific metric columns, collapsed into one generic 'baseline' display column.
_METRIC_COLS = ("auroc", "auprc", "rmse", "r2")


def _baseline_cell(row) -> str:
    """One generic baseline cell, task-aware: 'AUROC/AUPRC' for classification rows,
    'RMSE/R²' for regression rows (so a mixed --task all table needs no empty columns)."""
    if row.get("task") == "regression":
        rmse, r2 = row.get("rmse"), row.get("r2")
        if _is_missing(rmse) and _is_missing(r2):
            return "[dim]-[/dim]"
        rmse_s = f"{float(rmse):.3f}" if not _is_missing(rmse) else "-"
        r2_s = _metric_cell("r2", r2) if not _is_missing(r2) else "-"
        return f"{rmse_s}/{r2_s} [dim]RMSE/R²[/dim]"
    auroc, auprc = row.get("auroc"), row.get("auprc")
    if _is_missing(auroc) and _is_missing(auprc):
        return "[dim]-[/dim]"
    a = _metric_cell("auroc", auroc) if not _is_missing(auroc) else "-"
    p = _metric_cell("auprc", auprc) if not _is_missing(auprc) else "-"
    return f"{a}/{p} [dim]AUROC/AUPRC[/dim]"


# Ordered display spec: (df-column-key-or-None, header, justify, render(row)).
# `key` is what --sort_by sees; rendering reads whatever fields it needs from the row.
_RENDERERS = {
    "id": lambda r: f"[dim]{escape(str(r['id']))}[/dim]",
    "name": lambda r: f"[cyan]{escape(str(r['name']))}[/cyan]",
    "source": lambda r: _source_cell("source", r.get("source")),
    "size": lambda r: _size_cell(r.get("size")),
    "balance": _balance_cell,
    "baseline": _baseline_cell,
    "leaderboard": _leaderboard_cell,
}

# Display-only merges: each group of raw df columns collapses to one task-aware column.
_MERGES = {
    "balance": ("ratio", "skew"),                       # class balance OR target skew
    "baseline": _METRIC_COLS,                            # auroc/auprc OR rmse/r2
    "leaderboard": ("leaderboard_score", "leaderboard_metric"),
}
_MERGE_OF = {col: key for key, group in _MERGES.items() for col in group}
_HIDE = frozenset()  # every df column is shown

# Compact, color-coded task tags (nicer than the long "classification"/"regression").
_TASK_TAGS = {"classification": "[cyan]cls[/cyan]", "regression": "[magenta]reg[/magenta]"}


def _task_cell(v) -> str:
    return _TASK_TAGS.get(v, "[dim]-[/dim]" if _is_missing(v) else escape(str(v)))


_RENDERERS["task"] = lambda r: _task_cell(r.get("task"))


def _available_cell(v) -> str:
    return "[green]✓[/green]" if v else "[red]✗[/red]"


_RENDERERS["available"] = lambda r: _available_cell(r.get("available"))


def _availability_map(df) -> dict:
    """Live S3 HEAD-probe of every distinct (source, task, name) family in ``df``.

    One probe per *family*, not per row — a multi-column family (e.g. toxcast's 617
    columns under --expand) would otherwise fire the same check hundreds of times. Run
    concurrently since this is a live network call per family.
    """
    keys = sorted(set(zip(df["source"], df["task"], df["name"])))
    if not keys:
        return {}
    with ThreadPoolExecutor(max_workers=min(16, len(keys))) as pool:
        results = pool.map(lambda k: check_availability(*k), keys)
    return dict(zip(keys, results))


def _display_columns(df):
    """Build the ordered list of (header, justify, render(row)) for the given frame.

    Collapses task-specific raw columns into single task-aware display columns — class
    metrics + regression metrics → 'baseline'; ratio + skew → 'balance'; the leaderboard
    score/metric pair → 'leaderboard' — so a mixed classification/regression table is
    uniform with no empty cells. The DataFrame keeps all raw columns (so --sort_by and the
    API are unchanged).
    """
    spec = []
    seen = set()
    for col in df.columns:
        if col in _HIDE:
            continue
        key = _MERGE_OF.get(col)
        if key is not None:
            if key not in seen:
                spec.append((key, "left", _RENDERERS[key]))
                seen.add(key)
            continue
        header = DISPLAY_NAMES.get(col, col)
        justify = "right" if col in NUMERIC_COLUMNS else "left"
        render = _RENDERERS.get(col, (lambda c: lambda r: _plain_cell(c, r.get(c)))(col))
        spec.append((header, justify, render))

    # Place the label-shape cue (balance) just before the baseline metrics.
    headers = [h for h, _, _ in spec]
    if "baseline" in headers and "balance" in headers:
        baseline = spec.pop(headers.index("baseline"))
        bidx = [h for h, _, _ in spec].index("balance")
        spec.insert(bidx + 1, baseline)
    return spec


def _caption(full, shown, args) -> str:
    """A one-line summary of the view: counts + active sort + limit note."""
    if "n_columns" in full.columns:
        n_cols = int(full["n_columns"].fillna(0).sum())
        parts = [f"{len(full)} families · {n_cols} label columns"]
    else:
        parts = [f"{len(full)} task rows"]
    if args.sort_by:
        parts.append(f"sorted by {args.sort_by} {'↓' if args.desc else '↑'}")
    if args.limit is not None and len(full) > len(shown):
        parts.append(f"showing top {len(shown)} of {len(full)}")
    return "  ·  ".join(parts)


_EPILOG = """\
columns (which appear depends on --task; --sort_by key in parentheses):
  dataset       Dataset (family) identifier, e.g. "ames", "sider". (key: name)
  source        Where the data comes from: tdcommons or moleculenet.
  task          Task type: classification or regression.
  columns       Number of label columns (endpoints) in the family; 1 for
                single-label sets. (key: n_columns) [default view only]
  column        Name of the individual label column / endpoint. [--expand only]
  n_tot         Total samples; for multi-column families, the molecule count
                (abbreviated as 302k / 1.2M at/above 100,000).
  n_pos         Positive-class samples.                [classification only]
  ratio         Positive-class fraction, n_pos / n_tot, shown as a small bar +
                value. [classification only]
  auroc, auprc  Baseline AUROC / AUPRC (color-graded green/yellow/red). [classification]
  rmse, r2      Baseline RMSE / R-squared.             [regression]
  leaderboard   Best published score + its metric, e.g. "0.871 AUROC", where known.
                (sort with --sort_by leaderboard_score)
  last_updated  Date the dataset entry was last refreshed (YYYY-MM-DD).
  on S3         Green check / red cross: whether the dataset is present on the public
                S3 bucket right now. [--check_availability only]

The auroc/auprc/rmse/r2 columns are a RandomForest baseline averaged over random
K-fold cross-validation — a reference floor, not the best published model. The
leaderboard column is that best published model, where known (MoleculeNet, and TDC
ADMET tasks via Polaris); blank otherwise. Colors are shown on a terminal and
dropped when output is piped.

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
  eosbench catalog --name cyp --check_availability   also show live S3 availability

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
        help="Only this source, e.g. tdcommons, moleculenet (default: all).",
    )
    selection.add_argument(
        "--task",
        type=str,
        default="all",
        metavar="TASK",
        help="Task type: all, classification, or regression (default: all).",
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

    live = parser.add_argument_group("live checks")
    live.add_argument(
        "--check_availability",
        action="store_true",
        help="Live-check whether each shown dataset is present on the public S3 bucket "
        "yet, adding an 'on S3' column (green check / red cross). One HTTP HEAD request "
        "per dataset family shown (deduplicated, run concurrently) — needs network access "
        "and adds latency; off by default.",
    )

    args = parser.parse_args()

    df = get_catalog(source=args.source, task=args.task, expand=args.expand)
    try:
        # Filter + sort the full frame; apply --limit afterwards so the caption can
        # report how many rows were hidden.
        full = filter_catalog(
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
            limit=None,
        )
    except ValueError as e:
        parser.error(str(e))

    shown = full.head(args.limit) if args.limit is not None else full

    if args.check_availability and not shown.empty:
        n_families = shown[["source", "task", "name"]].drop_duplicates().shape[0]
        print(f"Checking S3 availability for {n_families} dataset(s)...", file=sys.stderr)
        avail_map = _availability_map(shown)
        shown = shown.copy()
        shown["available"] = [
            avail_map[key] for key in zip(shown["source"], shown["task"], shown["name"])
        ]

    spec = _display_columns(shown)
    table = Table(
        title="eosbench datasets",
        caption=_caption(full, shown, args),
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        title_style="bold",
        caption_style="dim",
    )
    for header, justify, _render in spec:
        # Keep every column on one line (ellipsize if cramped); only the dataset name —
        # the key identifier — is allowed to wrap so it's never silently truncated.
        table.add_column(
            header,
            justify=justify,
            no_wrap=(header != "dataset"),
            overflow="ellipsis",
        )
    for _, row in shown.iterrows():
        table.add_row(*[render(row) for _, _, render in spec])

    Console().print(table)

    if shown.empty:
        print("No datasets matched the given filters.", file=sys.stderr)


if __name__ == "__main__":
    main()
