import pandas as pd
import pytest

from src.train_caption_augmented_qwen import (
    build_training_records,
    captions_for_row,
    is_truthy,
    target_text_from_answer,
)


def make_row() -> pd.Series:
    return pd.Series(
        {
            "Id": "sample-1",
            "Input_1": "a.jpg",
            "Input_2": "b.jpg",
            "Input_3": "c.jpg",
            "Input_4": "d.jpg",
            "Sentence": "A person opens a box and takes out a cup.",
            "Answer": "[3, 2, 4, 1]",
            "No_ordering": "False",
        }
    )


def test_target_text_from_answer_converts_submission_inverse_to_chronological_order():
    assert target_text_from_answer("[3, 2, 4, 1]") == "[4, 2, 1, 3]"


def test_captions_for_row_reads_cache_by_id_index_and_image_name():
    row = make_row()
    cache = {
        ("sample-1", 1, "a.jpg"): "first caption",
        ("sample-1", 2, "b.jpg"): "second caption",
        ("sample-1", 3, "c.jpg"): "third caption",
        ("sample-1", 4, "d.jpg"): "fourth caption",
    }
    assert captions_for_row(row, cache, "fail") == [
        "first caption",
        "second caption",
        "third caption",
        "fourth caption",
    ]


def test_captions_for_row_can_fail_on_missing_caption():
    with pytest.raises(ValueError, match="Missing cached captions"):
        captions_for_row(make_row(), {}, "fail")


def test_captions_for_row_can_use_empty_missing_captions():
    assert captions_for_row(make_row(), {}, "empty") == ["", "", "", ""]


def test_is_truthy_handles_train_no_ordering_values():
    assert is_truthy("True")
    assert is_truthy("1")
    assert not is_truthy("False")
    assert not is_truthy("0")


def test_build_training_records_supports_dry_run_without_caption_cache(tmp_path):
    data_dir = tmp_path
    row = make_row()
    pd.DataFrame([row.to_dict()]).to_csv(data_dir / "train.csv", index=False)

    records = build_training_records(
        data_dir=data_dir,
        caption_cache_path=None,
        missing_caption_policy="empty",
        max_samples=1,
        drop_no_ordering=False,
    )

    assert len(records) == 1
    assert records[0].target_text == "[4, 2, 1, 3]"
    assert records[0].captions == ["", "", "", ""]
