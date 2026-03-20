import argparse

from rich.console import Console
from rich.table import Table

from ..dataset import DatasetInfo


def main():
    parser = argparse.ArgumentParser(description="Show metadata for a single eosbench dataset.")
    parser.add_argument("--source", type=str, required=True, help="Dataset source: tdc or chembl.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. ames.")
    parser.add_argument("--task", type=str, default="classification", help="Task type (default: classification).")
    args = parser.parse_args()

    info = DatasetInfo(source=args.source, task_type=args.task, dataset=args.dataset)
    meta = info.metadata

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
        ("auroc",        f"{meta['auroc_mean']:.4f} ± {meta['auroc_std']:.4f}" if "auroc_mean" in meta else "-"),
        ("aupr",         f"{meta['aupr_mean']:.4f} ± {meta['aupr_std']:.4f}" if "aupr_mean" in meta else "-"),
    ]
    for field, value in fields:
        table.add_row(field, value)

    Console().print(table)


if __name__ == "__main__":
    main()
