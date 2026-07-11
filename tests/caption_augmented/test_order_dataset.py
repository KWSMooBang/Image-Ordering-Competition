import pandas as pd
import pytest

from src.caption_augmented.captions import CaptionRecord, append_caption_record
from src.caption_augmented.dataset import (
    OrderTrainingDataset,
    build_training_records,
    is_truthy,
    target_text_from_answer,
)
from src.submission import parse_answer_cell


def make_train_row(**overrides):
    row = {
        "Id": "sample-1",
        "Input_1": "a.jpg",
        "Input_2": "b.jpg",
        "Input_3": "c.jpg",
        "Input_4": "d.jpg",
        "Sentence": "A person opens a box and takes out a cup.",
        "Answer": "[3, 2, 4, 1]",
        "No_ordering": "False",
    }
    row.update(overrides)
    return row


def test_target_text_from_answer_converts_submission_inverse_to_chronological_order():
    assert target_text_from_answer("[3, 2, 4, 1]") == "[4, 2, 1, 3]"


def test_is_truthy_handles_no_ordering_values():
    assert is_truthy("True")
    assert is_truthy("1")
    assert not is_truthy("False")
    assert not is_truthy("0")


def test_build_training_records_uses_caption_cache_and_target_conversion(tmp_path):
    data_dir = tmp_path
    pd.DataFrame([make_train_row()]).to_csv(data_dir / "train.csv", index=False)
    cache_path = tmp_path / "captions.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        append_caption_record(handle, CaptionRecord("sample-1", 1, "a.jpg", "caption 1"))
        append_caption_record(handle, CaptionRecord("sample-1", 2, "b.jpg", "caption 2"))
        append_caption_record(handle, CaptionRecord("sample-1", 3, "c.jpg", "caption 3"))
        append_caption_record(handle, CaptionRecord("sample-1", 4, "d.jpg", "caption 4"))

    records = build_training_records(
        data_dir,
        cache_path,
        missing_caption_policy="fail",
        max_samples=1,
        drop_no_ordering=False,
    )

    assert len(records) == 1
    assert records[0].captions == ["caption 1", "caption 2", "caption 3", "caption 4"]
    assert records[0].target_text == "[4, 2, 1, 3]"
    assert OrderTrainingDataset(records)[0] == records[0]


def test_build_training_records_can_drop_no_ordering_rows(tmp_path):
    data_dir = tmp_path
    pd.DataFrame([make_train_row(No_ordering="True")]).to_csv(data_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="No caption-augmented order training records"):
        build_training_records(
            data_dir,
            caption_cache_path=None,
            missing_caption_policy="empty",
            max_samples=None,
            drop_no_ordering=True,
        )


def test_build_training_records_can_read_filtered_train_csv(tmp_path):
    data_dir = tmp_path
    pd.DataFrame([make_train_row(Id="unfiltered")]).to_csv(data_dir / "train.csv", index=False)
    filtered_csv = tmp_path / "outputs" / "train_filtered.csv"
    filtered_csv.parent.mkdir(parents=True)
    pd.DataFrame([make_train_row(Id="filtered")]).to_csv(filtered_csv, index=False)

    records = build_training_records(
        data_dir,
        caption_cache_path=None,
        missing_caption_policy="empty",
        max_samples=None,
        drop_no_ordering=False,
        train_csv_path=filtered_csv,
    )

    assert [record.row["Id"] for record in records] == ["filtered"]


def test_build_training_records_can_shuffle_image_order_and_recompute_target(tmp_path):
    data_dir = tmp_path
    row = make_train_row(Answer="[3, 1, 4, 2]")
    pd.DataFrame([row]).to_csv(data_dir / "train.csv", index=False)
    cache_path = tmp_path / "captions.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        append_caption_record(handle, CaptionRecord("sample-1", 1, "a.jpg", "caption a"))
        append_caption_record(handle, CaptionRecord("sample-1", 2, "b.jpg", "caption b"))
        append_caption_record(handle, CaptionRecord("sample-1", 3, "c.jpg", "caption c"))
        append_caption_record(handle, CaptionRecord("sample-1", 4, "d.jpg", "caption d"))

    records = build_training_records(
        data_dir,
        caption_cache_path=cache_path,
        missing_caption_policy="fail",
        max_samples=None,
        drop_no_ordering=False,
        shuffle_augmentations_per_sample=1,
        shuffle_seed=3,
    )

    assert len(records) == 1
    shuffled_inputs = [records[0].row[f"Input_{index}"] for index in range(1, 5)]
    assert shuffled_inputs != ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert records[0].captions == [
        {
            "a.jpg": "caption a",
            "b.jpg": "caption b",
            "c.jpg": "caption c",
            "d.jpg": "caption d",
        }[image]
        for image in shuffled_inputs
    ]
    assert sorted(parse_answer_cell(records[0].target_text)) == [1, 2, 3, 4]


def test_build_training_records_can_keep_original_plus_shuffled_views(tmp_path):
    data_dir = tmp_path
    pd.DataFrame([make_train_row()]).to_csv(data_dir / "train.csv", index=False)

    records = build_training_records(
        data_dir,
        caption_cache_path=None,
        missing_caption_policy="empty",
        max_samples=None,
        drop_no_ordering=False,
        shuffle_augmentations_per_sample=2,
        shuffle_seed=3,
        shuffle_keep_original=True,
    )

    assert len(records) == 3
    assert [records[0].row[f"Input_{index}"] for index in range(1, 5)] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
