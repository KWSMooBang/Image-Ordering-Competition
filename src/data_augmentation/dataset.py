from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
import random
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS
from src.submission import PERMUTATION, format_answer, normalize_permutation, parse_answer_cell

IDENTITY_PERMUTATION = tuple(PERMUTATION)
SHUFFLE_PERMUTATIONS = tuple(tuple(order) for order in permutations(PERMUTATION))
NON_IDENTITY_SHUFFLE_PERMUTATIONS = tuple(order for order in SHUFFLE_PERMUTATIONS if order != IDENTITY_PERMUTATION)


@dataclass(frozen=True)
class RealtimeShuffleConfig:
    seed: int = 42
    augmentations_per_sample: int = 1
    include_identity: bool = False
    shuffle_no_ordering: bool = True

    def __post_init__(self) -> None:
        if self.augmentations_per_sample < 1:
            raise ValueError("augmentations_per_sample must be at least 1")


def _row_to_dict(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def recompute_answer_for_shuffle(answer: object, permutation: Sequence[int]) -> list[int]:
    """Recompute submission-style Answer after applying new_slot -> original_slot shuffle."""
    original_answer = parse_answer_cell(answer)
    shuffle = normalize_permutation(permutation)
    return [original_answer[original_slot - 1] for original_slot in shuffle]


def shuffle_row(
    row: Mapping[str, Any] | pd.Series,
    permutation: Sequence[int],
    *,
    id_suffix: str | None = None,
) -> dict[str, Any]:
    original_values = _row_to_dict(row)
    values = dict(original_values)
    shuffle = normalize_permutation(permutation)

    for new_slot, original_slot in enumerate(shuffle):
        values[INPUT_COLUMNS[new_slot]] = original_values[INPUT_COLUMNS[original_slot - 1]]

    if "Answer" in values:
        values["Answer"] = format_answer(recompute_answer_for_shuffle(original_values["Answer"], shuffle))

    if id_suffix is not None:
        values["Id"] = f"{values['Id']}{id_suffix}"

    return values


def sample_shuffle_permutation(
    rng: random.Random,
    *,
    include_identity: bool = False,
) -> list[int]:
    choices = SHUFFLE_PERMUTATIONS if include_identity else NON_IDENTITY_SHUFFLE_PERMUTATIONS
    return list(rng.choice(choices))


def _seed_for_item(seed: int, epoch: int, base_index: int, view_index: int) -> int:
    return seed + epoch * 1_000_003 + base_index * 10_007 + view_index * 101


class RealtimeShuffleDataset:
    """Map-style dataset that shuffles image slots at item access time.

    The dataset is intentionally framework-light: it works as a PyTorch map-style dataset without
    importing torch, and it is also easy to test as a plain Python sequence.
    """

    def __init__(
        self,
        rows: pd.DataFrame | Sequence[Mapping[str, Any]],
        config: RealtimeShuffleConfig | None = None,
    ) -> None:
        if isinstance(rows, pd.DataFrame):
            self._rows = [row.to_dict() for _, row in rows.iterrows()]
        else:
            self._rows = [dict(row) for row in rows]

        self.config = config or RealtimeShuffleConfig()
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._rows) * self.config.augmentations_per_sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        base_index, view_index = divmod(index, self.config.augmentations_per_sample)
        row = dict(self._rows[base_index])

        if not self.config.shuffle_no_ordering and _is_truthy(row.get("No_ordering")):
            return row

        item_seed = _seed_for_item(self.config.seed, self.epoch, base_index, view_index)
        permutation = sample_shuffle_permutation(
            random.Random(item_seed),
            include_identity=self.config.include_identity,
        )
        return shuffle_row(row, permutation)
