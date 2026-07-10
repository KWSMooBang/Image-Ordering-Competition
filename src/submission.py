from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PERMUTATION = [1, 2, 3, 4]
LIST_LITERAL_PATTERN = re.compile(r"\[[^\[\]]+\]")


def normalize_permutation(values: Iterable[int]) -> list[int]:
    result = [int(value) for value in values]
    if len(result) != 4 or sorted(result) != PERMUTATION:
        raise ValueError(f"Expected a permutation of {PERMUTATION}, got {result}")
    return result


def parse_answer_cell(value: object) -> list[int]:
    if isinstance(value, list):
        return normalize_permutation(value)
    if not isinstance(value, str):
        raise ValueError(f"Answer must be a list string, got {type(value).__name__}")
    parsed = ast.literal_eval(value)
    return normalize_permutation(parsed)


def format_answer(values: Sequence[int]) -> str:
    return str(normalize_permutation(values))


def chronological_to_submission(order: Sequence[int]) -> list[int]:
    chronological_order = normalize_permutation(order)
    answer = [0] * 4
    for position, image_number in enumerate(chronological_order, start=1):
        answer[image_number - 1] = position
    return answer


def submission_to_chronological(answer: Sequence[int]) -> list[int]:
    submission_answer = normalize_permutation(answer)
    order = [0] * 4
    for image_number, position in enumerate(submission_answer, start=1):
        order[position - 1] = image_number
    return order


def parse_permutation_from_text(text: str, fallback: Sequence[int] | None = None) -> list[int]:
    for match in LIST_LITERAL_PATTERN.finditer(text):
        try:
            return parse_answer_cell(match.group(0))
        except (SyntaxError, ValueError):
            continue

    if fallback is not None:
        return normalize_permutation(fallback)

    raise ValueError(f"Could not parse a permutation from model output: {text!r}")


def parse_model_output(output_text: str, fallback: Sequence[int] = PERMUTATION) -> list[int]:
    chronological_order = parse_permutation_from_text(output_text, fallback=fallback)
    return chronological_to_submission(chronological_order)


def validate_submission_frame(
    submission_df: pd.DataFrame,
    sample_df: pd.DataFrame | None = None,
) -> list[str]:
    errors: list[str] = []

    expected_columns = ["Id", "Answer"]
    if list(submission_df.columns) != expected_columns:
        errors.append(f"Submission columns must be {expected_columns}, got {list(submission_df.columns)}")
        return errors

    if submission_df["Id"].duplicated().any():
        duplicated = submission_df.loc[submission_df["Id"].duplicated(), "Id"].head().tolist()
        errors.append(f"Submission contains duplicated Id values, e.g. {duplicated}")

    for row_number, answer in enumerate(submission_df["Answer"], start=2):
        try:
            parse_answer_cell(answer)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"Invalid Answer at CSV line {row_number}: {answer!r} ({exc})")
            if len(errors) >= 10:
                errors.append("Stopped after 10 submission validation errors.")
                break

    if sample_df is not None:
        if len(submission_df) != len(sample_df):
            errors.append(f"Submission row count {len(submission_df)} != sample row count {len(sample_df)}")
        elif submission_df["Id"].tolist() != sample_df["Id"].tolist():
            errors.append("Submission Id order does not match sample_submission.csv")

    return errors


def write_submission(rows: list[dict[str, str]], output_path: str | Path, sample_df: pd.DataFrame | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame(rows, columns=["Id", "Answer"])
    errors = validate_submission_frame(submission_df, sample_df=sample_df)
    if errors:
        raise ValueError("\n".join(errors))
    submission_df.to_csv(output, index=False)
