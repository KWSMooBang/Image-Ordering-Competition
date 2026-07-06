from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.caption_augmented.captions import captions_for_row, load_caption_cache
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
) -> list[OrderTrainingRecord]:
    train_df = read_csv(data_dir / "train.csv")
    if drop_no_ordering:
        train_df = train_df[~train_df["No_ordering"].map(is_truthy)]
    if max_samples is not None:
        train_df = train_df.head(max_samples).copy()

    caption_cache = load_caption_cache(caption_cache_path) if caption_cache_path else {}
    records: list[OrderTrainingRecord] = []
    for _, row in train_df.iterrows():
        records.append(
            OrderTrainingRecord(
                row=row.to_dict(),
                captions=captions_for_row(row, caption_cache, missing_policy=missing_caption_policy),
                target_text=target_text_from_answer(row["Answer"]),
            )
        )

    if not records:
        raise ValueError("No caption-augmented order training records were built.")
    return records


class OrderTrainingDataset:
    def __init__(self, records: list[OrderTrainingRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OrderTrainingRecord:
        return self.records[index]
