from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_utils import read_csv
from src.submission import format_answer, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a valid identity-order submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/identity_submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sample_df = read_csv(data_dir / "sample_submission.csv")

    rows = [{"Id": row_id, "Answer": format_answer([1, 2, 3, 4])} for row_id in sample_df["Id"]]
    write_submission(rows, args.output, sample_df=sample_df)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
