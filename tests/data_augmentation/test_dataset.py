import pandas as pd
import pytest

from src.data_augmentation import (
    RealtimeShuffleConfig,
    RealtimeShuffleDataset,
    recompute_answer_for_shuffle,
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
        "Sentence": "A short event happens in order.",
        "Answer": "[3, 1, 4, 2]",
        "No_ordering": "False",
    }
    row.update(overrides)
    return row


def test_shuffle_row_reorders_inputs_and_recomputes_answer():
    augmented = shuffle_row(make_row(), [2, 4, 1, 3])

    assert [augmented[f"Input_{idx}"] for idx in range(1, 5)] == [
        "b.jpg",
        "d.jpg",
        "a.jpg",
        "c.jpg",
    ]
    assert parse_answer_cell(augmented["Answer"]) == [1, 2, 3, 4]


def test_recompute_answer_for_identity_returns_original_answer():
    assert recompute_answer_for_shuffle("[3, 1, 4, 2]", [1, 2, 3, 4]) == [3, 1, 4, 2]


def test_shuffle_row_can_add_id_suffix_for_static_exports():
    augmented = shuffle_row(make_row(), [1, 2, 3, 4], id_suffix="__shuffle_0")

    assert augmented["Id"] == "sample-1__shuffle_0"
    assert augmented["Input_1"] == "a.jpg"
    assert augmented["Answer"] == "[3, 1, 4, 2]"


def test_realtime_shuffle_dataset_is_deterministic_for_seed_and_epoch():
    rows = pd.DataFrame([make_row(), make_row(Id="sample-2")])
    config = RealtimeShuffleConfig(seed=7, augmentations_per_sample=2)

    first = RealtimeShuffleDataset(rows, config=config)
    second = RealtimeShuffleDataset(rows, config=config)

    assert len(first) == 4
    assert [first[index] for index in range(len(first))] == [second[index] for index in range(len(second))]


def test_realtime_shuffle_dataset_excludes_identity_by_default():
    dataset = RealtimeShuffleDataset([make_row()], config=RealtimeShuffleConfig(seed=3))
    augmented = dataset[0]

    assert [augmented[f"Input_{idx}"] for idx in range(1, 5)] != ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert sorted(parse_answer_cell(augmented["Answer"])) == [1, 2, 3, 4]


def test_realtime_shuffle_dataset_can_skip_no_ordering_rows():
    row = make_row(No_ordering="True")
    dataset = RealtimeShuffleDataset(
        [row],
        config=RealtimeShuffleConfig(seed=3, shuffle_no_ordering=False),
    )

    assert dataset[0] == row


def test_realtime_shuffle_dataset_rejects_invalid_virtual_length():
    with pytest.raises(ValueError, match="augmentations_per_sample"):
        RealtimeShuffleConfig(augmentations_per_sample=0)
