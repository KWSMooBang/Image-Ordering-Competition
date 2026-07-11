from pathlib import Path

import pandas as pd
from PIL import Image
from types import SimpleNamespace

from src.data_filtering import (
    DataFilteringConfig,
    FrameRelevanceScores,
    analyze_frame,
    analyze_sample,
    build_audit_frame,
    filter_train_frame,
)
from src.data_filtering.siglip import _pooled_tensor


def make_row(**overrides):
    row = {
        "Id": "sample-1",
        "Input_1": "a.jpg",
        "Input_2": "b.jpg",
        "Input_3": "c.jpg",
        "Input_4": "d.jpg",
        "Sentence": "A person opens a box and lifts a cup.",
        "Answer": "[3, 2, 4, 1]",
        "No_ordering": "False",
    }
    row.update(overrides)
    return row


def write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def write_pattern_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32))
    for y in range(32):
        for x in range(32):
            image.putpixel(
                (x, y),
                (
                    (color[0] + x * 5 + y * 3) % 256,
                    (color[1] + x * 2 + y * 7) % 256,
                    (color[2] + x * 11 + y) % 256,
                ),
            )
    image.save(path)


def write_sample_images(root: Path, row: dict):
    sample_dir = root / row["Id"]
    write_pattern_image(sample_dir / row["Input_1"], (10, 80, 140))
    write_pattern_image(sample_dir / row["Input_2"], (30, 120, 170))
    write_pattern_image(sample_dir / row["Input_3"], (90, 40, 110))
    write_pattern_image(sample_dir / row["Input_4"], (150, 180, 80))


def test_analyze_frame_detects_dark_and_bright_blank_frames(tmp_path):
    dark_path = tmp_path / "dark.jpg"
    bright_path = tmp_path / "bright.jpg"
    write_image(dark_path, (0, 0, 0))
    write_image(bright_path, (255, 255, 255))

    assert analyze_frame(dark_path, image_index=1).blank_kind == "dark"
    assert analyze_frame(bright_path, image_index=2).blank_kind == "bright"


def test_analyze_sample_drops_no_ordering_placeholder_rows(tmp_path):
    row = make_row(No_ordering="True", Answer="[1, 2, 3, 4]")
    write_sample_images(tmp_path, row)

    audit = analyze_sample(row, tmp_path)

    assert audit.action == "drop_from_supervised"
    assert audit.sample_weight == 0.0
    assert "no_ordering" in audit.reasons
    assert "identity_answer_for_no_ordering" in audit.reasons


def test_analyze_sample_downweights_single_blank_frame(tmp_path):
    row = make_row()
    write_sample_images(tmp_path, row)
    write_image(tmp_path / row["Id"] / row["Input_3"], (0, 0, 0))

    audit = analyze_sample(row, tmp_path)

    assert audit.action == "downweight"
    assert audit.manual_review
    assert audit.blank_frames == [3]
    assert "blank_frame" in audit.reasons


def test_analyze_sample_drops_multiple_blank_frames(tmp_path):
    row = make_row()
    write_sample_images(tmp_path, row)
    write_image(tmp_path / row["Id"] / row["Input_1"], (0, 0, 0))
    write_image(tmp_path / row["Id"] / row["Input_4"], (255, 255, 255))

    audit = analyze_sample(row, tmp_path)

    assert audit.action == "drop_from_supervised"
    assert "multiple_blank_frames" in audit.reasons


def test_caption_cache_alone_does_not_enable_lexical_relevance_filtering(tmp_path):
    row = make_row()
    write_sample_images(tmp_path, row)
    cache = {
        ("sample-1", 1, "a.jpg"): "a person with a cup",
        ("sample-1", 2, "b.jpg"): "a car parked on a street",
        ("sample-1", 3, "c.jpg"): "a cup near a box",
        ("sample-1", 4, "d.jpg"): "a person opens something",
    }

    audit = analyze_sample(row, tmp_path, caption_cache=cache)

    assert audit.relevance_backend == "none"
    assert audit.relevance_scores == []
    assert audit.caption_embedding_scores == []
    assert audit.low_relevance_frames == []
    assert "low_text_frame_relevance" not in audit.reasons


def test_analyze_sample_can_use_siglip_caption_embedding_scores(tmp_path):
    row = make_row()
    write_sample_images(tmp_path, row)
    cache = {
        ("sample-1", 1, "a.jpg"): "a person with a cup",
        ("sample-1", 2, "b.jpg"): "a car parked on a street",
        ("sample-1", 3, "c.jpg"): "a cup near a box",
        ("sample-1", 4, "d.jpg"): "a person opens something",
    }

    def scorer(row_values, image_paths, captions):
        assert row_values["Id"] == "sample-1"
        assert len(image_paths) == 4
        assert captions == [
            "a person with a cup",
            "a car parked on a street",
            "a cup near a box",
            "a person opens something",
        ]
        return FrameRelevanceScores(
            relevance_scores=[0.44, 0.03, 0.51, 0.27],
            image_relevance_scores=[0.44, 0.92, 0.51, 0.27],
            caption_embedding_scores=[0.76, 0.03, 0.81, 0.62],
        )

    audit = analyze_sample(
        row,
        tmp_path,
        caption_cache=cache,
        relevance_scorer=scorer,
        relevance_backend="siglip",
    )

    assert audit.action == "downweight"
    assert audit.relevance_backend == "siglip"
    assert audit.relevance_scores == [0.44, 0.03, 0.51, 0.27]
    assert audit.image_relevance_scores == [0.44, 0.92, 0.51, 0.27]
    assert audit.caption_embedding_scores == [0.76, 0.03, 0.81, 0.62]
    assert audit.low_relevance_frames == [2]
    assert "low_text_frame_relevance" in audit.reasons


def test_siglip_pooled_tensor_supports_transformers_output_object():
    output = SimpleNamespace(pooler_output="pooled")

    assert _pooled_tensor(output) == "pooled"


def test_build_audit_frame_and_filter_train_frame_preserve_id_order(tmp_path):
    row_keep = make_row(Id="keep")
    row_drop = make_row(Id="drop", No_ordering="True", Answer="[1, 2, 3, 4]")
    for row in [row_keep, row_drop]:
        write_sample_images(tmp_path, row)
    train_df = pd.DataFrame([row_keep, row_drop])

    audit_df = build_audit_frame(train_df, tmp_path, config=DataFilteringConfig())
    filtered = filter_train_frame(train_df, audit_df)

    assert audit_df["Id"].tolist() == ["keep", "drop"]
    assert filtered["Id"].tolist() == ["keep"]
