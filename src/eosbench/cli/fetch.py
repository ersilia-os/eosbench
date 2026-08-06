import argparse
import sys

from ..dataset import FEATURIZATIONS, get_catalog, mirror_dataset, resolve_id
from ..utils.logging import logger
from .catalog import filter_catalog


def _select_datasets(source, task, name):
    """Return ``[(source, dataset, task), ...]``, one entry per matching family."""
    df = get_catalog(source=source, task=task, expand=False)
    df = filter_catalog(df, name=name)
    return list(zip(df["source"], df["name"], df["task"]))


def _fetch_many(targets, *, featurization, output_dir, from_dir):
    """Mirror every target, logging and continuing past failures.

    Returns the dataset names that failed.
    """
    failures = []
    for i, (source, dataset, task) in enumerate(targets, 1):
        logger.info(f"[{i}/{len(targets)}] Fetching {source}/{dataset}...")
        try:
            mirror_dataset(
                source=source,
                dataset=dataset,
                featurization=featurization,
                output_dir=output_dir,
                task=task,
                from_dir=from_dir,
            )
        except RuntimeError as e:
            logger.error(f"[{dataset}] {e}")
            failures.append(dataset)
    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``eosbench fetch`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Download an eosbench dataset to a local folder."
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="eosbench identifier (resolves source/dataset/task automatically; "
        "the simplest way to fetch). Use instead of --source/--dataset.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Dataset source, e.g. tdcommons or moleculenet (not needed with --id).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name, e.g. ames (not needed with --id).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch every dataset matching --source/--task/--name instead of a single one.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Filter by name substring (only with --all).",
    )
    parser.add_argument(
        "--featurization",
        type=str,
        default="morgan",
        help=f"Featurization to download: {', '.join(FEATURIZATIONS)} or none (default: morgan).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Root output directory (default: .).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task type (default: classification).",
    )
    parser.add_argument(
        "--from-dir",
        dest="from_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Copy files from this local directory (laid out as "
        "DIR/{source}/{task}/{dataset}/) instead of downloading from S3.",
    )
    return parser


def _run_fetch_all(parser, args) -> None:
    """Handle ``--all``: fetch every dataset matching the source/task/name filters."""
    if args.id or args.dataset:
        parser.error("--all cannot be combined with --id or --dataset.")
    targets = _select_datasets(source=args.source, task=args.task, name=args.name)
    if not targets:
        logger.warning("No datasets matched the given filters.")
        return
    featurization = None if args.featurization.lower() == "none" else args.featurization
    failures = _fetch_many(
        targets,
        featurization=featurization,
        output_dir=args.output_dir,
        from_dir=args.from_dir,
    )
    logger.info(f"Fetched {len(targets) - len(failures)}/{len(targets)} datasets.")
    if failures:
        sys.exit(1)


def _run_fetch_one(parser, args) -> None:
    """Handle the default case: fetch the single dataset identified by --id or --source/--dataset."""
    if args.name is not None:
        parser.error("--name only applies together with --all.")

    source, dataset, task = args.source, args.dataset, args.task
    if args.id is not None:
        try:
            hit = resolve_id(args.id)
        except KeyError as e:
            parser.error(str(e))
        source, dataset, task = hit["source"], hit["dataset"], hit["task"]
    elif not (source and dataset):
        parser.error("provide --id, or both --source and --dataset.")

    featurization = None if args.featurization.lower() == "none" else args.featurization
    try:
        dest = mirror_dataset(
            source=source,
            dataset=dataset,
            featurization=featurization,
            output_dir=args.output_dir,
            task=task,
            from_dir=args.from_dir,
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.success(f"Dataset saved to {dest}")


def main():
    """Entry point for ``eosbench fetch``: parse args and mirror the requested dataset(s)."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.all:
        _run_fetch_all(parser, args)
    else:
        _run_fetch_one(parser, args)


if __name__ == "__main__":
    main()
