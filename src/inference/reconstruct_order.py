"""Reconstruct a complete four-image order from six pair probabilities."""

from __future__ import annotations

import argparse
import ast
import math
from itertools import combinations, permutations
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def _parse_pair_key(key: object) -> tuple[int, int]:
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[0]), int(key[1])

    if isinstance(key, str):
        cleaned = key.strip().replace("(", "").replace(")", "")
        for separator in ("_", ",", "-", ":"):
            if separator in cleaned:
                first, second = cleaned.split(separator, maxsplit=1)
                return int(first.strip()), int(second.strip())

    raise ValueError(f"Unsupported pair key: {key!r}")


def normalise_pair_probabilities(
    pair_probabilities: Mapping[object, float],
) -> dict[tuple[int, int], float]:
    normalised: dict[tuple[int, int], float] = {}
    for raw_key, raw_probability in pair_probabilities.items():
        first, second = _parse_pair_key(raw_key)
        if first == second:
            raise ValueError("A pair cannot compare an image with itself.")

        probability = float(raw_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1]; got {probability}.")

        if first < second:
            key = (first, second)
            canonical_probability = probability
        else:
            key = (second, first)
            canonical_probability = 1.0 - probability

        if key in normalised and not math.isclose(
            normalised[key], canonical_probability, abs_tol=1e-6
        ):
            raise ValueError(f"Conflicting probabilities supplied for pair {key}.")

        normalised[key] = canonical_probability

    return normalised


def probability_before(
    pair_probabilities: Mapping[tuple[int, int], float],
    first: int,
    second: int,
) -> float:
    if first < second:
        return pair_probabilities[(first, second)]
    return 1.0 - pair_probabilities[(second, first)]


def score_order(
    order: Sequence[int],
    pair_probabilities: Mapping[tuple[int, int], float],
    *,
    epsilon: float = 1e-7,
) -> float:
    score = 0.0
    for left_position in range(len(order)):
        for right_position in range(left_position + 1, len(order)):
            probability = probability_before(
                pair_probabilities,
                order[left_position],
                order[right_position],
            )
            probability = min(max(probability, epsilon), 1.0 - epsilon)
            score += math.log(probability)
    return score


def reconstruct_best_order(
    pair_probabilities: Mapping[object, float],
    *,
    num_items: int = 4,
) -> tuple[list[int], float]:
    probabilities = normalise_pair_probabilities(pair_probabilities)
    expected_pairs = set(combinations(range(1, num_items + 1), 2))
    missing_pairs = expected_pairs - set(probabilities)
    if missing_pairs:
        raise KeyError(f"Missing probabilities for pairs: {sorted(missing_pairs)}")

    candidates = permutations(range(1, num_items + 1))
    best_order, best_score = max(
        ((list(order), score_order(order, probabilities)) for order in candidates),
        key=lambda item: item[1],
    )
    return best_order, best_score


def order_to_answer(order: Sequence[int]) -> list[int]:
    expected = list(range(1, len(order) + 1))
    if sorted(order) != expected:
        raise ValueError(f"order must be a permutation of {expected}; got {list(order)}.")

    answer = [0] * len(order)
    for chronological_position, image_index in enumerate(order, start=1):
        answer[image_index - 1] = chronological_position
    return answer


def reconstruct_dataframe(
    pair_dataframe: pd.DataFrame,
    *,
    probability_column: str = "probability",
) -> pd.DataFrame:
    required = {
        "sample_id",
        "image_a_index",
        "image_b_index",
        probability_column,
    }
    missing = sorted(required - set(pair_dataframe.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    records: list[dict] = []
    for sample_id, group in pair_dataframe.groupby("sample_id", sort=False):
        pair_probabilities = {
            (int(row.image_a_index), int(row.image_b_index)): float(
                getattr(row, probability_column)
            )
            for row in group.itertuples(index=False)
        }
        order, score = reconstruct_best_order(pair_probabilities)
        records.append(
            {
                "Id": str(sample_id),
                "order": str(order),
                "Answer": str(order_to_answer(order)),
                "order_score": score,
            }
        )

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct full image orders from pair probabilities."
    )
    parser.add_argument("--input", required=True, help="Pair prediction CSV")
    parser.add_argument("--output", required=True, help="Output reconstructed CSV")
    parser.add_argument("--probability-column", default="probability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    pair_dataframe = pd.read_csv(input_path)
    reconstructed = reconstruct_dataframe(
        pair_dataframe,
        probability_column=args.probability_column,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_csv(output_path, index=False)
    print(f"[reconstruct] samples={len(reconstructed):,}")
    print(f"[reconstruct] saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
