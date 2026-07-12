import pandas as pd

from src.constrained_likelihood_tta.augmentation import (
    build_augmented_rows,
    recompute_answer,
    shuffle_row,
)
from src.submission import parse_answer_cell


def make_row(**overrides):
    row = {
        "Id": "sample-1",
        "Input_1": "a.jpg",
        "Input_2": "b.jpg",
        "Input_3": "c.jpg",
        "Input_4": "d.jpg",
        "Sentence": "A short chronological story.",
        "Answer": "[1, 2, 3, 4]",
        "No_ordering": True,
    }
    row.update(overrides)
    return row


def test_shuffle_row_reorders_inputs_and_answer():
    row = shuffle_row(make_row(Answer="[3, 1, 4, 2]"), [2, 4, 1, 3])

    assert [row[f"Input_{index}"] for index in range(1, 5)] == [
        "b.jpg",
        "d.jpg",
        "a.jpg",
        "c.jpg",
    ]
    assert parse_answer_cell(row["Answer"]) == [1, 2, 3, 4]
    assert recompute_answer("[3, 1, 4, 2]", [2, 4, 1, 3]) == [1, 2, 3, 4]


def test_augmentation_shuffles_no_ordering_identity_rows_too():
    rows = build_augmented_rows(
        pd.DataFrame([make_row()]),
        augmentations_per_sample=2,
        seed=3,
        keep_original=True,
    )

    assert len(rows) == 3
    assert rows[0]["Answer"] == "[1, 2, 3, 4]"
    for row in rows[1:]:
        assert [row[f"Input_{index}"] for index in range(1, 5)] != [
            "a.jpg",
            "b.jpg",
            "c.jpg",
            "d.jpg",
        ]
        assert sorted(parse_answer_cell(row["Answer"])) == [1, 2, 3, 4]
