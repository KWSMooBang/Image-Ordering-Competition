from pathlib import Path

import pandas as pd
import pytest

from src.caption_augmented.prompts import build_caption_prompt, build_order_messages


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


def test_build_caption_prompt_is_sentence_aware_without_order_guessing():
    prompt = build_caption_prompt(make_row(), image_index=2)

    assert "Story sentence" in prompt
    assert "Image 2" in prompt
    assert "Do not guess the final image order" in prompt


def test_build_order_messages_interleaves_images_and_captions():
    row = make_row()
    captions = ["first", "second", "third", "fourth"]
    messages = build_order_messages(row, image_dir=Path("/data/test"), captions=captions)
    content = messages[0]["content"]

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
    assert "chronological order" in text


def test_build_order_messages_requires_four_captions():
    with pytest.raises(ValueError, match="Expected 4 captions"):
        build_order_messages(make_row(), image_dir=Path("/data/test"), captions=["only one"])
