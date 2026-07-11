from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import permutations
import random
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS
from src.submission import PERMUTATION, normalize_permutation

IDENTITY_PERMUTATION = tuple(PERMUTATION)
ALL_PERMUTATIONS = tuple(tuple(order) for order in permutations(PERMUTATION))
BALANCED_PERMUTATIONS = (
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
    """Return deterministic, position-balanced permutations with identity first."""
    if count < 1 or count > len(ALL_PERMUTATIONS):
        raise ValueError(
            f"TTA permutation count must be between 1 and {len(ALL_PERMUTATIONS)}, got {count}"
        )

    selected = list(BALANCED_PERMUTATIONS[:count])
    if count > len(selected):
        remaining = [order for order in ALL_PERMUTATIONS if order not in BALANCED_PERMUTATIONS]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return [list(order) for order in selected]


def permute_row_and_captions(
    row: Mapping[str, Any] | pd.Series,
    captions: Sequence[str],
    permutation: Sequence[int],
) -> tuple[pd.Series, list[str]]:
    """Reorder image slots and captions using a new-slot-to-original-slot permutation."""
    if len(captions) != len(INPUT_COLUMNS):
        raise ValueError(f"Expected {len(INPUT_COLUMNS)} captions, got {len(captions)}")

    order = normalize_permutation(permutation)
    original = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    permuted = dict(original)
    for new_index, original_slot in enumerate(order):
        permuted[INPUT_COLUMNS[new_index]] = original[INPUT_COLUMNS[original_slot - 1]]

    permuted_captions = [str(captions[original_slot - 1]) for original_slot in order]
    return pd.Series(permuted), permuted_captions


def restore_chronological_order(
    permuted_order: Sequence[int],
    permutation: Sequence[int],
) -> list[int]:
    """Map chronological labels from permuted slots back to original image labels."""
    chronological = normalize_permutation(permuted_order)
    new_slot_to_original = normalize_permutation(permutation)
    return [new_slot_to_original[new_slot - 1] for new_slot in chronological]


def consensus_chronological_order(orders: Sequence[Sequence[int]]) -> tuple[list[int], int]:
    """Choose the majority order, breaking ties by earliest TTA view."""
    if not orders:
        raise ValueError("At least one valid chronological order is required for TTA voting")

    normalized = [tuple(normalize_permutation(order)) for order in orders]
    counts = Counter(normalized)
    winner = max(counts, key=lambda order: (counts[order], -normalized.index(order)))
    return list(winner), counts[winner]
