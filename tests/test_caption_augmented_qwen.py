from pathlib import Path

import pandas as pd

from src.caption_augmented_common import (
    append_caption_cache,
    clean_caption,
    get_caption_message,
    get_order_message,
    load_caption_cache,
)
from src.qwen_vl_common import DEFAULT_MODEL_NAME, get_processor_kwargs, load_peft_adapter


def make_row() -> pd.Series:
    return pd.Series(
        {
            "Id": "sample-1",
            "Input_1": "a.jpg",
            "Input_2": "b.jpg",
            "Input_3": "c.jpg",
            "Input_4": "d.jpg",
            "Sentence": "A person opens a box and takes out a cup.",
        }
    )


def test_clean_caption_trims_quotes_and_truncates_on_word_boundary():
    caption = clean_caption('  "A person opens the cardboard box carefully."  ', max_chars=100)
    assert caption == "A person opens the cardboard box carefully."

    long_caption = clean_caption("alpha beta gamma delta", max_chars=14)
    assert long_caption == "alpha beta..."


def test_default_model_targets_qwen3_vl_8b():
    assert DEFAULT_MODEL_NAME == "Qwen/Qwen3-VL-8B-Instruct"


def test_processor_kwargs_include_only_configured_pixel_budgets():
    assert get_processor_kwargs(min_pixels=None, max_pixels=None) == {}
    assert get_processor_kwargs(min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28) == {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }


def test_load_peft_adapter_noops_without_adapter_path():
    model = object()
    assert load_peft_adapter(model, None) is model


def test_caption_message_targets_one_image_without_order_guess():
    row = make_row()
    message = get_caption_message(row, image_path=Path("/tmp/sample-1/a.jpg"), image_index=1)

    content = message[0]["content"]
    assert content[0] == {"type": "image", "image": "/tmp/sample-1/a.jpg"}
    assert "Do not guess the final image order" in content[1]["text"]
    assert row["Sentence"] in content[1]["text"]


def test_order_message_interleaves_images_and_captions():
    row = make_row()
    captions = ["caption 1", "caption 2", "caption 3", "caption 4"]
    message = get_order_message(row, image_dir=Path("/data/test"), captions=captions)
    content = message[0]["content"]

    image_items = [item for item in content if item["type"] == "image"]
    text = "\n".join(item["text"] for item in content if item["type"] == "text")
    assert [item["image"] for item in image_items] == [
        "/data/test/sample-1/a.jpg",
        "/data/test/sample-1/b.jpg",
        "/data/test/sample-1/c.jpg",
        "/data/test/sample-1/d.jpg",
    ]
    for index, caption in enumerate(captions, start=1):
        assert f"Image {index} caption: {caption}" in text
    assert "chronological image labels" in text


def test_caption_cache_round_trips_latest_record(tmp_path):
    cache_path = tmp_path / "captions.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        append_caption_cache(handle, "sample-1", 1, "a.jpg", "old caption")
        append_caption_cache(handle, "sample-1", 1, "a.jpg", "new caption")

    cache = load_caption_cache(cache_path)
    assert cache[("sample-1", 1, "a.jpg")] == "new caption"
