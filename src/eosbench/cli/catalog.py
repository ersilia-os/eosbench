import argparse

from rich.console import Console
from rich.table import Table

from ..dataset import get_catalog


def main():
    parser = argparse.ArgumentParser(description="List available eosbench datasets.")
    parser.add_argument("--source", type=str, default=None, help="Filter by source: tdc or chembl.")
    parser.add_argument("--task", type=str, default="classification", help="Task type (default: classification).")
    args = parser.parse_args()

    df = get_catalog(source=args.source, task_type=args.task)

    table = Table(title="eosbench datasets")
    for col in df.columns:
        table.add_column(col, style="cyan" if col == "name" else None)

    for _, row in df.iterrows():
        table.add_row(*[
            f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")
            for v in row
        ])

    Console().print(table)


if __name__ == "__main__":
    main()
