import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..dataset import check_availability, get_catalog

# Columns that may appear in a get_catalog() frame (collapsed or expand=True,
# classification or regression). --sort-by is validated against the actual
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
    "leaderboard_split",
    "leaderboard_provider",
    "leaderboard_comparable",
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
# column names (used by --sort-by and the library API) are unchanged.
DISPLAY_NAMES = {
    "name": "dataset",
    "n_columns": "columns",
    "baseline": "baseline [random split]",
    "leaderboard_score": "lb_score",
    "leaderboard_metric": "lb_metric",
    "leaderboard_split": "lb_split",
    "leaderboard_provider": "lb_provider",
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

    Parameters
    ----------
    df : pandas.DataFrame
        A ``get_catalog()`` frame.
    name : str or None
        Case-insensitive substring filter on the dataset name.
    min_samples, max_samples : int or None
        Inclusive bounds on ``n_tot``.
    min_ratio, max_ratio : float or None
        Inclusive bounds on ``ratio`` (classification).
    min_auroc, max_auroc, min_auprc, max_auprc : float or None
        Inclusive bounds on ``auroc``/``auprc`` (classification).
    min_rmse, max_rmse, min_r2, max_r2 : float or None
        Inclusive bounds on ``rmse``/``r2`` (regression).
    sort_by : str or None
        Column to sort by.
    desc : bool, default False
        Sort descending instead of ascending.
    limit : int or None
        Keep only the first ``limit`` rows after sorting.

    Returns
    -------
    pandas.DataFrame
    """
    if name is not None:
        df = df[df["name"].str.contains(name, case=False, na=False)]
    df = _threshold(df, "n_tot", min_samples, ">=", "--min-samples")
    df = _threshold(df, "n_tot", max_samples, "<=", "--max-samples")
    df = _threshold(df, "ratio", min_ratio, ">=", "--min-ratio")
    df = _threshold(df, "ratio", max_ratio, "<=", "--max-ratio")
    df = _threshold(df, "auroc", min_auroc, ">=", "--min-auroc")
    df = _threshold(df, "auroc", max_auroc, "<=", "--max-auroc")
    df = _threshold(df, "auprc", min_auprc, ">=", "--min-auprc")
    df = _threshold(df, "auprc", max_auprc, "<=", "--max-auprc")
    df = _threshold(df, "rmse", min_rmse, ">=", "--min-rmse")
    df = _threshold(df, "rmse", max_rmse, "<=", "--max-rmse")
    df = _threshold(df, "r2", min_r2, ">=", "--min-r2")
    df = _threshold(df, "r2", max_r2, "<=", "--max-r2")

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
    """The regression analog of the class-balance cue. Skewness is unbounded (unlike ratio,
    naturally in [0, 1]), so the bar is center-anchored instead of left-anchored: the center
    cell is always filled as a "symmetric/balanced" baseline (▱▱▰▱▱ by default), and the two
    cells on the skewed side fill outward from center as |skew| grows, saturating at |skew|>=2.
    Same glyph set as the ratio bar (no separate divider glyph) for visual consistency."""
    if _is_missing(v):
        return "[dim]-[/dim]"
    s = float(v)
    extra = round(min(abs(s) / 2.0, 1.0) * 2)  # 0, 1, or 2 cells beyond the center
    if s < 0:
        left, right = "▱" * (2 - extra) + "▰" * extra, "▱▱"
    else:
        left, right = "▱▱", "▰" * extra + "▱" * (2 - extra)
    return f"[dim]{left}▰{right}[/dim] {s:+.2f}"


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


# Whether a local single-run scaffold-holdout result is a fair comparison to this
# leaderboard score (see leaderboard_comparable's docstring in dataset.get_catalog).
_COMPARABLE_GLYPHS = {
    "yes": "[green]✓[/green]",
    "split_only": "[yellow]±[/yellow]",
    "no": "[red]✗[/red]",
    "unverified": "[dim]?[/dim]",
}


def _leaderboard_cell(row) -> str:
    """Merge leaderboard_score + leaderboard_metric + leaderboard_comparable into one
    cell: '0.871 AUROC ±' -- the trailing glyph is leaderboard_comparable (✓ same
    split & single-run, ± same split but a multi-run average, ✗ different
    split/test-set, ? unverified)."""
    score, metric = row.get("leaderboard_score"), row.get("leaderboard_metric")
    if _is_missing(score):
        return "[dim]-[/dim]"
    cell = f"[{_grade_color(float(score))}]{float(score):.3f}[/]"
    if not _is_missing(metric):
        cell += f" [dim]{escape(str(metric))}[/dim]"
    glyph = _COMPARABLE_GLYPHS.get(row.get("leaderboard_comparable"))
    if glyph:
        cell += f" {glyph}"
    return cell


def _source_cell(_col, v) -> str:
    return (
        "[dim]-[/dim]" if _is_missing(v) else f"[{_source_color(v)}]{escape(str(v))}[/]"
    )


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
# `key` is what --sort-by sees; rendering reads whatever fields it needs from the row.
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
    "balance": ("ratio", "skew"),  # class balance OR target skew
    "baseline": _METRIC_COLS,  # auroc/auprc OR rmse/r2
    # leaderboard_comparable renders as a trailing glyph on this cell rather than its own
    # column (see _leaderboard_cell) -- kept out of the table as raw text either way.
    "leaderboard": (
        "leaderboard_score",
        "leaderboard_metric",
        "leaderboard_comparable",
    ),
}
_MERGE_OF = {col: key for key, group in _MERGES.items() for col in group}
_HIDE = frozenset()  # every df column is shown

# Compact, color-coded task tags (nicer than the long "classification"/"regression").
_TASK_TAGS = {
    "classification": "[cyan]cls[/cyan]",
    "regression": "[magenta]reg[/magenta]",
}


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
    uniform with no empty cells. The DataFrame keeps all raw columns (so --sort-by and the
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
                spec.append((DISPLAY_NAMES.get(key, key), "left", _RENDERERS[key]))
                seen.add(key)
            continue
        header = DISPLAY_NAMES.get(col, col)
        justify = "right" if col in NUMERIC_COLUMNS else "left"
        render = _RENDERERS.get(
            col, (lambda c: lambda r: _plain_cell(c, r.get(c)))(col)
        )
        spec.append((header, justify, render))

    # Place the label-shape cue (balance) just before the baseline metrics.
    baseline_header = DISPLAY_NAMES.get("baseline", "baseline")
    balance_header = DISPLAY_NAMES.get("balance", "balance")
    headers = [h for h, _, _ in spec]
    if baseline_header in headers and balance_header in headers:
        baseline = spec.pop(headers.index(baseline_header))
        bidx = [h for h, _, _ in spec].index(balance_header)
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
columns (which appear depends on --task; --sort-by key in parentheses):
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
                (sort with --sort-by leaderboard_score)
  lb_split      What split that published score was computed on (e.g. "scaffold",
                "random"). Only directly comparable to this row's own scaffold split
                when lb_provider is this same source (e.g. tdcommons via "polaris",
                which mirrors TDC's own ADMET Benchmark Group) -- when lb_provider is a
                different source (e.g. "moleculenet"), lb_split describes *that other
                dataset's* split, not this one's.
  lb_provider   Where the published score came from: this source's own leaderboard, or
                cross-filled from a different one covering the same assay (e.g. a
                tdcommons row citing a MoleculeNet number) -- see lb_split above.
  last_updated  Date the dataset entry was last refreshed (YYYY-MM-DD).
  on S3         Green check / red cross: whether the dataset is present on the public
                S3 bucket right now. [--check-availability only]

The auroc/auprc/rmse/r2 columns are a RandomForest baseline averaged over random
K-fold cross-validation — a reference floor, not the best published model. The
leaderboard column is that best published model, where known (MoleculeNet, and TDC
ADMET tasks via Polaris); blank otherwise. Colors are shown on a terminal and
dropped when output is piped.

examples:
  eosbench catalog                                  list every dataset
  eosbench catalog --task regression --sort-by rmse  regression sets, lowest RMSE first
  eosbench catalog --name cyp                        datasets whose name contains "cyp"
  eosbench catalog --min-samples 1000 --max-ratio 0.5
                                                    big, imbalanced datasets
  eosbench catalog --min-auroc 0.8 --sort-by auroc --desc
                                                    strong baselines, best first
  eosbench catalog --sort-by n_tot --desc --limit 5  the 5 largest datasets
  eosbench catalog --expand                          one row per label column (multi-column sets)
  eosbench catalog --name cyp --check-availability   also show live S3 availability

Threshold filters drop datasets with a missing value for that column (e.g.
--min-auroc skips datasets with no recorded AUROC). Filters combine with AND.
"""


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the "dataset selection" argument group (``--source``/``--task``/``--expand``)."""
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


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the "filters (combine with AND)" argument group (the min/max threshold flags)."""
    filters = parser.add_argument_group("filters (combine with AND)")
    filters.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="SUBSTR",
        help="Keep datasets whose name contains SUBSTR (case-insensitive).",
    )
    filters.add_argument(
        "--min-samples",
        type=int,
        default=None,
        metavar="N",
        help="Keep datasets with n_tot >= N.",
    )
    filters.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Keep datasets with n_tot <= N.",
    )
    filters.add_argument(
        "--min-ratio",
        type=float,
        default=None,
        metavar="R",
        help="[classification] Keep datasets with positive-class ratio >= R.",
    )
    filters.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        metavar="R",
        help="[classification] Keep datasets with positive-class ratio <= R.",
    )
    filters.add_argument(
        "--min-auroc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUROC >= A.",
    )
    filters.add_argument(
        "--max-auroc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUROC <= A.",
    )
    filters.add_argument(
        "--min-auprc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUPRC >= A.",
    )
    filters.add_argument(
        "--max-auprc",
        type=float,
        default=None,
        metavar="A",
        help="[classification] Keep datasets with baseline AUPRC <= A.",
    )
    filters.add_argument(
        "--min-rmse",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline RMSE >= V.",
    )
    filters.add_argument(
        "--max-rmse",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline RMSE <= V.",
    )
    filters.add_argument(
        "--min-r2",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline R-squared >= V.",
    )
    filters.add_argument(
        "--max-r2",
        type=float,
        default=None,
        metavar="V",
        help="[regression] Keep datasets with baseline R-squared <= V.",
    )


def _add_ordering_args(parser: argparse.ArgumentParser) -> None:
    """Add the "sorting and limiting" argument group (``--sort-by``/``--desc``/``--limit``)."""
    ordering = parser.add_argument_group("sorting and limiting")
    ordering.add_argument(
        "--sort-by",
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


def _add_live_check_args(parser: argparse.ArgumentParser) -> None:
    """Add the "live checks" argument group (``--check-availability``)."""
    live = parser.add_argument_group("live checks")
    live.add_argument(
        "--check-availability",
        action="store_true",
        help="Live-check whether each shown dataset is present on the public S3 bucket "
        "yet, adding an 'on S3' column (green check / red cross). One HTTP HEAD request "
        "per dataset family shown (deduplicated, run concurrently) — needs network access "
        "and adds latency; off by default.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``eosbench catalog`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="eosbench catalog",
        description="List available eosbench datasets as a table, with optional "
        "filtering, sorting and limiting.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selection_args(parser)
    _add_filter_args(parser)
    _add_ordering_args(parser)
    _add_live_check_args(parser)
    return parser


def _filter_and_sort(parser, args) -> pd.DataFrame:
    """Load the catalog and apply the requested filters/sort (unlimited).

    Exits via ``parser.error`` on an invalid filter (e.g. a metric column absent
    for the current task).
    """
    df = get_catalog(source=args.source, task=args.task, expand=args.expand)
    try:
        return filter_catalog(
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


def _add_availability_column(shown: pd.DataFrame) -> pd.DataFrame:
    """Live-check S3 availability and add the resulting ``available`` column."""
    n_families = shown[["source", "task", "name"]].drop_duplicates().shape[0]
    print(f"Checking S3 availability for {n_families} dataset(s)...", file=sys.stderr)
    avail_map = _availability_map(shown)
    shown = shown.copy()
    shown["available"] = [
        avail_map[key] for key in zip(shown["source"], shown["task"], shown["name"])
    ]
    return shown


def _render_table(full, shown, args) -> None:
    """Render ``shown`` as a Rich table with a caption summarizing ``full`` vs ``shown``."""
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
        # the key identifier — is allowed to wrap so it's never silently truncated. Escape
        # the header text itself: Rich treats it as markup, so a literal "[...]" in a
        # DISPLAY_NAMES label (e.g. "baseline [random split]") would otherwise be parsed
        # as an (invalid, silently-dropped) style tag instead of shown verbatim.
        table.add_column(
            escape(header),
            justify=justify,
            no_wrap=(header != "dataset"),
            overflow="ellipsis",
        )
    for _, row in shown.iterrows():
        table.add_row(*[render(row) for _, _, render in spec])

    Console().print(table)

    if shown.empty:
        print("No datasets matched the given filters.", file=sys.stderr)


def main():
    """Entry point for ``eosbench catalog``: parse args and print the filtered table."""
    parser = _build_parser()
    args = parser.parse_args()

    full = _filter_and_sort(parser, args)
    shown = full.head(args.limit) if args.limit is not None else full

    if args.check_availability and not shown.empty:
        shown = _add_availability_column(shown)

    _render_table(full, shown, args)


if __name__ == "__main__":
    main()
