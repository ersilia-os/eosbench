import argparse

from ..dataset import mirror_dataset, FEATURIZATIONS
from ..utils.logging import logger


def main():
    parser = argparse.ArgumentParser(description="Download an eosbench dataset to a local folder.")
    parser.add_argument("--source", type=str, required=True, help="Dataset source: tdc or chembl.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. ames.")
    parser.add_argument(
        "--featurization", type=str, default="morgan",
        help=f"Featurization to download: {', '.join(FEATURIZATIONS)} or none (default: morgan).",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Root output directory (default: .).")
    parser.add_argument("--task", type=str, default="classification", help="Task type (default: classification).")
    args = parser.parse_args()

    featurization = None if args.featurization.lower() == "none" else args.featurization

    dest = mirror_dataset(
        source=args.source,
        dataset=args.dataset,
        featurization=featurization,
        output_dir=args.output_dir,
        task_type=args.task,
    )
    logger.success(f"Dataset saved to {dest}")


if __name__ == "__main__":
    main()
