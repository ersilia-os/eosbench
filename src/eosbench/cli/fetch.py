import argparse
import sys

from ..dataset import mirror_dataset, resolve_id, FEATURIZATIONS
from ..utils.logging import logger


def main():
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
        "--dataset", type=str, default=None, help="Dataset name, e.g. ames (not needed with --id)."
    )
    parser.add_argument(
        "--featurization",
        type=str,
        default="morgan",
        help=f"Featurization to download: {', '.join(FEATURIZATIONS)} or none (default: morgan).",
    )
    parser.add_argument(
        "--output_dir",
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
        "--from_dir",
        dest="from_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Copy files from this local directory (laid out as "
        "DIR/{source}/{task}/{dataset}/) instead of downloading from S3.",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
