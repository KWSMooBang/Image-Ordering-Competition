from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.submission import parse_answer_cell

INPUT_COLUMNS = ["Input_1", "Input_2", "Input_3", "Input_4"]
TRAIN_COLUMNS = ["Id", *INPUT_COLUMNS, "Sentence", "Answer", "No_ordering"]
TEST_COLUMNS = ["Id", *INPUT_COLUMNS, "Sentence"]
SUBMISSION_COLUMNS = ["Id", "Answer"]


@dataclass(frozen=True)
class DataSummary:
    train_rows: int
    test_rows: int
    sample_rows: int
    checked_image_paths: int


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def image_paths_for_row(row: pd.Series, image_root: Path) -> list[Path]:
    return [image_root / str(row["Id"]) / str(row[column]) for column in INPUT_COLUMNS]


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> list[str]:
    if list(df.columns) != expected:
        return [f"{name} columns must be {expected}, got {list(df.columns)}"]
    return []


def _iter_missing_images(df: pd.DataFrame, image_root: Path, limit: int | None = None) -> Iterable[str]:
    checked = 0
    for _, row in df.iterrows():
        for path in image_paths_for_row(row, image_root):
            checked += 1
            if not path.exists():
                yield str(path)
        if limit is not None and checked >= limit:
            return


def validate_data_dir(data_dir: str | Path, image_check_limit: int | None = None) -> tuple[DataSummary, list[str], list[str]]:
    root = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    sample_csv = root / "sample_submission.csv"
    train_dir = root / "train"
    test_dir = root / "test"

    for path in [train_csv, test_csv, sample_csv, train_dir, test_dir]:
        if not path.exists():
            errors.append(f"Missing required path: {path}")

    if errors:
        return DataSummary(0, 0, 0, 0), errors, warnings

    train_df = read_csv(train_csv)
    test_df = read_csv(test_csv)
    sample_df = read_csv(sample_csv)

    errors.extend(_check_columns(train_df, TRAIN_COLUMNS, "train.csv"))
    errors.extend(_check_columns(test_df, TEST_COLUMNS, "test.csv"))
    errors.extend(_check_columns(sample_df, SUBMISSION_COLUMNS, "sample_submission.csv"))

    if test_df["Id"].tolist() != sample_df["Id"].tolist():
        errors.append("test.csv Id order does not match sample_submission.csv")

    if train_df["Id"].duplicated().any():
        errors.append("train.csv contains duplicated Id values")
    if test_df["Id"].duplicated().any():
        errors.append("test.csv contains duplicated Id values")

    invalid_answers = []
    for line_number, value in enumerate(train_df.get("Answer", []), start=2):
        try:
            parse_answer_cell(value)
        except (SyntaxError, ValueError) as exc:
            invalid_answers.append(f"line {line_number}: {value!r} ({exc})")
            if len(invalid_answers) >= 5:
                break
    if invalid_answers:
        errors.append("Invalid train Answer values: " + "; ".join(invalid_answers))

    missing_images = []
    for missing in _iter_missing_images(train_df, train_dir, limit=image_check_limit):
        missing_images.append(missing)
        if len(missing_images) >= 10:
            break
    for missing in _iter_missing_images(test_df, test_dir, limit=image_check_limit):
        missing_images.append(missing)
        if len(missing_images) >= 10:
            break
    if missing_images:
        errors.append("Missing image files, e.g. " + "; ".join(missing_images))

    no_ordering_values = set(str(value) for value in train_df["No_ordering"].dropna().unique())
    if not no_ordering_values.issubset({"True", "False", "true", "false", "0", "1"}):
        warnings.append(f"Unexpected No_ordering values: {sorted(no_ordering_values)}")

    checked_images = 4 * (len(train_df) + len(test_df))
    if image_check_limit is not None:
        checked_images = min(checked_images, image_check_limit)

    summary = DataSummary(
        train_rows=len(train_df),
        test_rows=len(test_df),
        sample_rows=len(sample_df),
        checked_image_paths=checked_images,
    )
    return summary, errors, warnings
