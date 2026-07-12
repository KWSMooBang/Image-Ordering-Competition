from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations
import random
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS
from src.submission import PERMUTATION, normalize_permutation

ALL_TTA_PERMUTATIONS = tuple(tuple(order) for order in permutations(PERMUTATION))
BALANCED_TTA_PERMUTATIONS = (
    (1, 2, 3, 4),
    (2, 3, 4, 1),
    (3, 4, 1, 2),
    (4, 1, 2, 3),
    (4, 3, 2, 1),
    (3, 2, 1, 4),
    (2, 1, 4, 3),
    (1, 4, 3, 2),
)


def build_tta_permutations(count: int, *, seed: int = 42) -> list[list[int]]:
    if count < 1 or count > len(ALL_TTA_PERMUTATIONS):
        raise ValueError("TTA permutation count must be between 1 and 24")
    selected = list(BALANCED_TTA_PERMUTATIONS[:count])
    if count > len(selected):
        remaining = [
            order
            for order in ALL_TTA_PERMUTATIONS
            if order not in BALANCED_TTA_PERMUTATIONS
        ]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return [list(order) for order in selected]


def permute_row_and_captions(
    row: Mapping[str, Any] | pd.Series,
    captions: Sequence[str],
    permutation: Sequence[int],
) -> tuple[pd.Series, list[str]]:
    if len(captions) != len(INPUT_COLUMNS):
        raise ValueError(f"Expected {len(INPUT_COLUMNS)} captions, got {len(captions)}")
    original = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    values = dict(original)
    new_slot_to_original = normalize_permutation(permutation)
    for new_slot, original_slot in enumerate(new_slot_to_original):
        values[INPUT_COLUMNS[new_slot]] = original[INPUT_COLUMNS[original_slot - 1]]
    permuted_captions = [
        str(captions[original_slot - 1]) for original_slot in new_slot_to_original
    ]
    return pd.Series(values), permuted_captions
