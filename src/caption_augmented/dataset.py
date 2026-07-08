from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.caption_augmented.captions import captions_for_row, load_caption_cache
from src.data_augmentation import RealtimeShuffleConfig, RealtimeShuffleDataset
from src.data_utils import read_csv
from src.submission import parse_answer_cell, submission_to_chronological


@dataclass(frozen=True)
class OrderTrainingRecord:
    row: dict[str, Any]
    captions: list[str]
    target_text: str


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def target_text_from_answer(answer: object) -> str:
    submission_answer = parse_answer_cell(answer)
    chronological_order = submission_to_chronological(submission_answer)
    return str(chronological_order)


def build_training_records(
    data_dir: Path,
    caption_cache_path: str | Path | None,
    *,
    missing_caption_policy: str,
    max_samples: int | None,
    drop_no_ordering: bool,
    train_csv_path: str | Path | None = None,
    shuffle_augmentations_per_sample: int = 0,
    shuffle_seed: int = 42,
    shuffle_include_identity: bool = False,
    shuffle_no_ordering: bool = False,
    shuffle_keep_original: bool = False,
) -> list[OrderTrainingRecord]:
    train_df = read_csv(train_csv_path if train_csv_path is not None else data_dir / "train.csv")
    if drop_no_ordering:
        train_df = train_df[~train_df["No_ordering"].map(is_truthy)]
    if max_samples is not None:
        train_df = train_df.head(max_samples).copy()

    rows = _build_training_rows(
        train_df,
        shuffle_augmentations_per_sample=shuffle_augmentations_per_sample,
        shuffle_seed=shuffle_seed,
        shuffle_include_identity=shuffle_include_identity,
        shuffle_no_ordering=shuffle_no_ordering,
        shuffle_keep_original=shuffle_keep_original,
    )
    caption_cache = load_caption_cache(caption_cache_path) if caption_cache_path else {}
    records: list[OrderTrainingRecord] = []
    for row_values in rows:
        row = pd.Series(row_values)
        records.append(
            OrderTrainingRecord(
                row=row_values,
                captions=captions_for_row(row, caption_cache, missing_policy=missing_caption_policy),
                target_text=target_text_from_answer(row["Answer"]),
            )
        )

    if not records:
        raise ValueError("No caption-augmented order training records were built.")
    return records


def _build_training_rows(
    train_df: pd.DataFrame,
    *,
    shuffle_augmentations_per_sample: int,
    shuffle_seed: int,
    shuffle_include_identity: bool,
    shuffle_no_ordering: bool,
    shuffle_keep_original: bool,
) -> list[dict[str, Any]]:
    original_rows = [row.to_dict() for _, row in train_df.iterrows()]
    if shuffle_augmentations_per_sample <= 0:
        return original_rows

    dataset = RealtimeShuffleDataset(
        train_df,
        config=RealtimeShuffleConfig(
            seed=shuffle_seed,
            augmentations_per_sample=shuffle_augmentations_per_sample,
            include_identity=shuffle_include_identity,
            shuffle_no_ordering=shuffle_no_ordering,
        ),
    )
    shuffled_rows = [dataset[index] for index in range(len(dataset))]
    if shuffle_keep_original:
        return original_rows + shuffled_rows
    return shuffled_rows


class OrderTrainingDataset:
    def __init__(self, records: list[OrderTrainingRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OrderTrainingRecord:
        return self.records[index]
