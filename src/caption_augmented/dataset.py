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


def load_filtered_train_dataframe(
    data_dir: Path,
    train_csv_path: str | Path | None,
    *,
    drop_no_ordering: bool,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Load train.csv (or an override path, e.g. a data_filtering output) and
    apply the drop-no-ordering / max-samples filters shared by train and val."""
    train_df = read_csv(train_csv_path if train_csv_path is not None else data_dir / "train.csv")
    if drop_no_ordering:
        train_df = train_df[~train_df["No_ordering"].map(is_truthy)]
    if max_samples is not None:
        train_df = train_df.head(max_samples).copy()
    return train_df.reset_index(drop=True)


def split_train_val_dataframe(
    train_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split rows (each row is one independent sample/Id)
    into a train set and a held-out validation set. Shuffling is seeded so the
    same split is reproduced across runs/restarts given the same input frame.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be between 0 and 1 (exclusive), got {val_fraction}")
    if len(train_df) < 2:
        raise ValueError("Need at least 2 rows to carve out a validation split.")

    shuffled = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    train_split_df = shuffled.iloc[val_size:].reset_index(drop=True)
    return train_split_df, val_df


def build_records_from_dataframe(
    train_df: pd.DataFrame,
    caption_cache_path: str | Path | None,
    *,
    missing_caption_policy: str,
    shuffle_augmentations_per_sample: int = 0,
    shuffle_seed: int = 42,
    shuffle_include_identity: bool = False,
    shuffle_no_ordering: bool = False,
    shuffle_keep_original: bool = False,
) -> list[OrderTrainingRecord]:
    """Build training records from an already-loaded/filtered DataFrame. Shared
    by `build_training_records` (path-based, train side) and callers that pass
    an in-memory train-only split (e.g. after `split_train_val_dataframe`)."""
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


def build_validation_records(
    val_df: pd.DataFrame,
    caption_cache_path: str | Path | None,
    *,
    missing_caption_policy: str,
) -> list[OrderTrainingRecord]:
    """Build validation records: same shape as training records (row, captions,
    ground-truth chronological `target_text`), but never shuffle-augmented, so
    validation always measures performance on the canonical input-slot order.
    """
    return build_records_from_dataframe(
        val_df,
        caption_cache_path,
        missing_caption_policy=missing_caption_policy,
        shuffle_augmentations_per_sample=0,
    )


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
    train_df = load_filtered_train_dataframe(
        data_dir,
        train_csv_path,
        drop_no_ordering=drop_no_ordering,
        max_samples=max_samples,
    )
    return build_records_from_dataframe(
        train_df,
        caption_cache_path,
        missing_caption_policy=missing_caption_policy,
        shuffle_augmentations_per_sample=shuffle_augmentations_per_sample,
        shuffle_seed=shuffle_seed,
        shuffle_include_identity=shuffle_include_identity,
        shuffle_no_ordering=shuffle_no_ordering,
        shuffle_keep_original=shuffle_keep_original,
    )


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