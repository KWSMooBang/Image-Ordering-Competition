from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations
import random
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS
from src.submission import (
    PERMUTATION,
    format_answer,
    normalize_permutation,
    parse_answer_cell,
)

IDENTITY = tuple(PERMUTATION)
NON_IDENTITY_SHUFFLES = tuple(
    tuple(order) for order in permutations(PERMUTATION) if tuple(order) != IDENTITY
)


def recompute_answer(answer: object, shuffle: Sequence[int]) -> list[int]:
    original_answer = parse_answer_cell(answer)
    new_slot_to_original = normalize_permutation(shuffle)
    return [
        original_answer[original_slot - 1] for original_slot in new_slot_to_original
    ]


def shuffle_row(
    row: Mapping[str, Any] | pd.Series,
    shuffle: Sequence[int],
) -> dict[str, Any]:
    original = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    values = dict(original)
    new_slot_to_original = normalize_permutation(shuffle)
    for new_slot, original_slot in enumerate(new_slot_to_original):
        values[INPUT_COLUMNS[new_slot]] = original[INPUT_COLUMNS[original_slot - 1]]
    values["Answer"] = format_answer(
        recompute_answer(original["Answer"], new_slot_to_original)
    )
    return values


def build_augmented_rows(
    train_df: pd.DataFrame,
    *,
    augmentations_per_sample: int,
    seed: int,
    keep_original: bool,
) -> list[dict[str, Any]]:
    if augmentations_per_sample < 0:
        raise ValueError("augmentations_per_sample must be non-negative")

    original_rows = [row.to_dict() for _, row in train_df.iterrows()]
    if augmentations_per_sample == 0:
        return original_rows

    augmented: list[dict[str, Any]] = []
    for row_index, row in enumerate(original_rows):
        for view_index in range(augmentations_per_sample):
            item_seed = seed + row_index * 10_007 + view_index * 101
            shuffle = random.Random(item_seed).choice(NON_IDENTITY_SHUFFLES)
            augmented.append(shuffle_row(row, shuffle))
    return original_rows + augmented if keep_original else augmented
