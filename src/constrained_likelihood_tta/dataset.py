from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.constrained_likelihood_tta.augmentation import build_augmented_rows
from src.constrained_likelihood_tta.captions import captions_for_row, load_caption_cache
from src.data_utils import read_csv
from src.submission import parse_answer_cell, submission_to_chronological


@dataclass(frozen=True)
class OrderTrainingRecord:
    row: dict[str, Any]
    captions: list[str]
    target_text: str


def target_text_from_answer(answer: object) -> str:
    return str(submission_to_chronological(parse_answer_cell(answer)))


def build_training_records(
    *,
    data_dir: Path,
    train_csv_path: str | Path | None,
    caption_cache_path: str | Path | None,
    caption_missing_policy: str,
    max_samples: int | None,
    shuffle_augmentations_per_sample: int,
    shuffle_seed: int,
    shuffle_keep_original: bool,
) -> list[OrderTrainingRecord]:
    source = (
        Path(train_csv_path) if train_csv_path is not None else data_dir / "train.csv"
    )
    train_df = read_csv(source)
    if max_samples is not None:
        train_df = train_df.head(max_samples).copy()

    rows = build_augmented_rows(
        train_df,
        augmentations_per_sample=shuffle_augmentations_per_sample,
        seed=shuffle_seed,
        keep_original=shuffle_keep_original,
    )
    cache = load_caption_cache(caption_cache_path)
    records: list[OrderTrainingRecord] = []
    for row_values in rows:
        row = pd.Series(row_values)
        records.append(
            OrderTrainingRecord(
                row=row_values,
                captions=captions_for_row(
                    row, cache, missing_policy=caption_missing_policy
                ),
                target_text=target_text_from_answer(row["Answer"]),
            )
        )
    if not records:
        raise ValueError("No constrained likelihood training records were built")
    return records


class OrderTrainingDataset:
    def __init__(self, records: list[OrderTrainingRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OrderTrainingRecord:
        return self.records[index]
