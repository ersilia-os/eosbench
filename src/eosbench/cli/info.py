import argparse

from rich.console import Console
from rich.table import Table

from ..dataset import DatasetInfo

# Shown under the metric tables. Mirrors the wording in `eosbench catalog --help`.
_BASELINE_NOTE = (
    "[dim]auroc/auprc are a RandomForest baseline (a reference floor, not the best "
    "published model). 'random split' = mean over random K-fold cross-validation; "
    "'scaffold split' = a single Bemis–Murcko scaffold holdout.[/dim]"
)


def main():
    parser = argparse.ArgumentParser(description="Show metadata for a single eosbench dataset.")
    parser.add_argument("--source", type=str, required=True, help="Dataset source, e.g. tdcommons or moleculenet.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. ames.")
    parser.add_argument("--task", type=str, default="classification", help="Task type (default: classification).")
    parser.add_argument("--column", type=str, default=None, help="For a multi-column family, show full details for one label column (untruncated).")
    args = parser.parse_args()

    info = DatasetInfo(source=args.source, task=args.task, dataset=args.dataset)
    meta = info.metadata
    console = Console()

    if args.column is not None:
        columns = meta.get("columns")
        if not columns:
            parser.error(
                f"{args.dataset!r} is a single-column dataset; omit --column "
                f"(its full details are shown by `eosbench info` without --column)."
            )
        if args.column not in columns:
            parser.error(
                f"unknown column {args.column!r} for {args.source}/{args.dataset}. "
                f"Available: {', '.join(columns)}"
            )
        _print_column_detail(console, args.source, args.dataset, args.column, columns[args.column])
        console.print(_BASELINE_NOTE)
        return

    if meta.get("columns"):  # family format: summary + per-column table
        summary = Table(title=f"{args.source}/{args.dataset}", show_header=False)
        summary.add_column("field", style="bold cyan")
        summary.add_column("value")
        summary.add_row("source", str(meta.get("source", "-")))
        summary.add_row("dataset", str(meta.get("dataset", "-")))
        summary.add_row("task", args.task)
        summary.add_row("n_molecules", str(meta.get("n_molecules", "-")))
        summary.add_row("n_columns", str(len(meta["columns"])))
        for field, value in _leaderboard_rows(meta):
            summary.add_row(field, value)
        summary.add_row("last_updated", str(meta.get("last_updated", "-")))
        console.print(summary)

        per_column = Table(title="columns")
        for col in ("column", "n", "pos", "neg", "auroc (random split)", "auprc (random split)",
                    "auroc (scaffold split)", "description"):
            per_column.add_column(col)
        for name, c in meta["columns"].items():
            per_column.add_row(
                name,
                str(c.get("n_samples", "-")),
                str(c.get("n_positives", "-")),
                str(c.get("n_negatives", "-")),
                _pm(c.get("random_auroc_mean"), c.get("random_auroc_std")),
                _pm(c.get("random_aupr_mean"), c.get("random_aupr_std")),
                f"{c['scaffold_auroc']:.4f}" if c.get("scaffold_auroc") is not None else "-",
                _truncate(c.get("description"), 70),
            )
        console.print(per_column)
        console.print(
            f"[dim]descriptions truncated; run `eosbench info --source {args.source} "
            f"--dataset {args.dataset} --column <name>` for one column's full details.[/dim]"
        )
        console.print(_BASELINE_NOTE)
        return

    table = Table(title=f"{args.source}/{args.dataset}", show_header=False)
    table.add_column("field", style="bold cyan")
    table.add_column("value")
    fields = [
        ("source",       meta.get("source", "-")),
        ("dataset",      meta.get("dataset", "-")),
        ("task",         args.task),
        ("n_samples",    str(meta.get("n_samples", "-"))),
        ("n_positives",  str(meta.get("n_positives", "-"))),
        ("n_negatives",  str(meta.get("n_negatives", "-"))),
        ("auroc",        _pm(meta.get("auroc_mean"), meta.get("auroc_std"))),
        ("auprc",        _pm(meta.get("aupr_mean"), meta.get("aupr_std"))),
    ]
    fields += _leaderboard_rows(meta)
    fields += _description_rows(meta)
    for field, value in fields:
        table.add_row(field, value)

    console.print(table)
    console.print(_BASELINE_NOTE)


def _truncate(text, width: int) -> str:
    """Truncate a description for table display, tolerating missing values."""
    if not text:
        return "-"
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def _pm(mean, std) -> str:
    """Format 'mean ± std', tolerating missing values."""
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _leaderboard_rows(meta: dict) -> list[tuple[str, str]]:
    """Best-published-model rows, only when the dataset records a leaderboard entry."""
    value, metric = meta.get("leaderboard_value"), meta.get("leaderboard_metric")
    if value is None and metric is None:
        return []
    score = f"{value:.4f}" if isinstance(value, (int, float)) else "-"
    rows = [("leaderboard", f"{score} ({metric})" if metric else score)]
    if meta.get("leaderboard_split"):
        rows.append(("leaderboard_split", str(meta["leaderboard_split"])))
    if meta.get("leaderboard_provider"):
        rows.append(("leaderboard_provider", str(meta["leaderboard_provider"])))
    if meta.get("leaderboard_source"):
        rows.append(("leaderboard_source", str(meta["leaderboard_source"])))
    return rows


def _description_rows(meta: dict) -> list[tuple[str, str]]:
    """Full (untruncated) description rows, only when present."""
    rows = []
    if meta.get("description"):
        rows.append(("description", str(meta["description"])))
    if meta.get("description_source"):
        rows.append(("description_source", str(meta["description_source"])))
    return rows


def _count(v) -> str:
    """Grouped-integer count, tolerating missing values."""
    if v is None:
        return "-"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


# Keys rendered by curated rows below; any other key is dumped generically so
# fields added to the metadata later still show up in the --column view.
_COLUMN_CURATED_KEYS = {
    "n_samples", "n_positives", "n_negatives",
    "random_auroc_mean", "random_auroc_std", "random_aupr_mean", "random_aupr_std",
    "scaffold_auroc", "scaffold_aupr",
    "leaderboard_value", "leaderboard_metric", "leaderboard_split", "leaderboard_provider",
    "leaderboard_source",
    "description", "description_source",
}


def _column_detail_rows(c: dict) -> list[tuple[str, str]]:
    """Ordered (field, value) rows describing a single label column, full text."""
    rows: list[tuple[str, str]] = [
        ("n_samples", _count(c.get("n_samples"))),
        ("n_positives", _count(c.get("n_positives"))),
        ("n_negatives", _count(c.get("n_negatives"))),
    ]
    n_tot, n_pos = c.get("n_samples"), c.get("n_positives")
    if n_tot and n_pos is not None:
        rows.append(("ratio", f"{n_pos / n_tot:.4f}"))
    rows += [
        ("auroc (random split)", _pm(c.get("random_auroc_mean"), c.get("random_auroc_std"))),
        ("auprc (random split)", _pm(c.get("random_aupr_mean"), c.get("random_aupr_std"))),
    ]
    if c.get("scaffold_auroc") is not None:
        rows.append(("auroc (scaffold split)", f"{c['scaffold_auroc']:.4f}"))
    if c.get("scaffold_aupr") is not None:
        rows.append(("auprc (scaffold split)", f"{c['scaffold_aupr']:.4f}"))
    rows += _leaderboard_rows(c)
    rows += _description_rows(c)
    # Any remaining keys (e.g. fields added to the metadata later), shown verbatim.
    for key, value in c.items():
        if key not in _COLUMN_CURATED_KEYS and value is not None:
            rows.append((key, str(value)))
    return rows


def _print_column_detail(console, source: str, dataset: str, column: str, c: dict) -> None:
    """Vertical, untruncated detail table for one label column of a family."""
    table = Table(title=f"{source}/{dataset} — {column}", show_header=False)
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value")  # wraps full text; never truncated
    for field, value in _column_detail_rows(c):
        table.add_row(field, value)
    console.print(table)


if __name__ == "__main__":
    main()
