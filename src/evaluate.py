from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_utils import read_csv
from src.submission import parse_answer_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a prediction CSV against train.csv answers.")
    parser.add_argument("--truth", default="data/train.csv")
    parser.add_argument("--pred", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    truth_df = read_csv(Path(args.truth))[["Id", "Answer"]].rename(columns={"Answer": "truth"})
    pred_df = read_csv(Path(args.pred))[["Id", "Answer"]].rename(columns={"Answer": "pred"})
    merged = truth_df.merge(pred_df, on="Id", how="inner")

    if len(merged) == 0:
        print("No overlapping Id values between truth and predictions.")
        return 1

    exact = 0
    positions = 0
    total_positions = 4 * len(merged)

    for _, row in merged.iterrows():
        truth = parse_answer_cell(row["truth"])
        pred = parse_answer_cell(row["pred"])
        exact += int(truth == pred)
        positions += sum(int(a == b) for a, b in zip(truth, pred))

    print(f"Rows evaluated: {len(merged)}")
    print(f"Exact match accuracy: {exact / len(merged):.6f}")
    print(f"Position accuracy: {positions / total_positions:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
