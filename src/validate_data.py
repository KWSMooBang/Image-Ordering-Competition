from __future__ import annotations

import argparse
import sys

from src.data_utils import validate_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate competition data layout and CSV contracts.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv, test.csv, sample_submission.csv")
    parser.add_argument("--image-check-limit", type=int, default=None, help="Optional cap for image path checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, errors, warnings = validate_data_dir(args.data_dir, image_check_limit=args.image_check_limit)

    print(
        "Data summary: "
        f"train={summary.train_rows}, "
        f"test={summary.test_rows}, "
        f"sample_submission={summary.sample_rows}, "
        f"checked_images={summary.checked_image_paths}"
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Data validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
