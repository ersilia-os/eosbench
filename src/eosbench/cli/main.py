import argparse
import sys

from .catalog import main as catalog_main
from .fetch import main as fetch_main
from .info import main as info_main


def main():
    """Entry point for the ``eosbench`` command: dispatch to the ``catalog``/``info``/
    ``fetch`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="eosbench",
        description="eosbench — Ersilia benchmark dataset CLI",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    subparsers.add_parser("catalog", help="List available datasets.", add_help=False)
    subparsers.add_parser("info", help="Show metadata for a dataset.", add_help=False)
    subparsers.add_parser("fetch", help="Download a dataset to disk.", add_help=False)

    # Parse only the subcommand name; pass the rest to the subcommand.
    args, remaining = parser.parse_known_args()
    sys.argv = [f"eosbench {args.command}"] + remaining

    if args.command == "catalog":
        catalog_main()
    elif args.command == "info":
        info_main()
    elif args.command == "fetch":
        fetch_main()


if __name__ == "__main__":
    main()
