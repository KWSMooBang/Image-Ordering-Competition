import pandas as pd
import pytest

from src.caption_augmented.captions import CaptionRecord, append_caption_record
from src.caption_augmented.dataset import (
    OrderTrainingDataset,
    build_training_records,
    build_validation_records,
    is_truthy,
    load_filtered_train_dataframe,
    split_train_val_dataframe,
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


# ---------------------------------------------------------------------------
# Train/val split (used by train.py's --val-fraction)
# ---------------------------------------------------------------------------


def test_load_filtered_train_dataframe_applies_drop_no_ordering_and_max_samples(tmp_path):
    data_dir = tmp_path
    rows = [
        make_train_row(Id="keep-1"),
        make_train_row(Id="drop-me", No_ordering="True"),
        make_train_row(Id="keep-2"),
    ]
    pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)

    df = load_filtered_train_dataframe(data_dir, None, drop_no_ordering=True, max_samples=1)

    assert len(df) == 1
    assert df.iloc[0]["Id"] == "keep-1"


def test_split_train_val_dataframe_is_deterministic_and_covers_every_row():
    df = pd.DataFrame([make_train_row(Id=f"sample-{i}") for i in range(10)])

    train_a, val_a = split_train_val_dataframe(df, val_fraction=0.2, seed=7)
    train_b, val_b = split_train_val_dataframe(df, val_fraction=0.2, seed=7)

    assert len(val_a) == 2
    assert len(train_a) == 8
    # Same seed -> identical split.
    assert list(train_a["Id"]) == list(train_b["Id"])
    assert list(val_a["Id"]) == list(val_b["Id"])
    # No overlap, full coverage.
    assert set(train_a["Id"]) | set(val_a["Id"]) == set(df["Id"])
    assert set(train_a["Id"]) & set(val_a["Id"]) == set()


def test_split_train_val_dataframe_rejects_fraction_out_of_range():
    df = pd.DataFrame([make_train_row(Id=f"sample-{i}") for i in range(5)])
    with pytest.raises(ValueError, match="--val-fraction"):
        split_train_val_dataframe(df, val_fraction=0.0, seed=1)
    with pytest.raises(ValueError, match="--val-fraction"):
        split_train_val_dataframe(df, val_fraction=1.0, seed=1)


def test_build_validation_records_never_shuffles_even_if_asked():
    df = pd.DataFrame([make_train_row()])
    records = build_validation_records(df, caption_cache_path=None, missing_caption_policy="empty")

    assert len(records) == 1
    assert [records[0].row[f"Input_{index}"] for index in range(1, 5)] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert records[0].target_text == target_text_from_answer(df.iloc[0]["Answer"])