from pathlib import Path

import pandas as pd
import pytest

from src.caption_augmented.captions import (
    CaptionRecord,
    append_caption_record,
    captions_for_row,
    clean_caption,
    default_caption_cache_path,
    generate_captions_for_row,
    load_caption_cache,
)


class FakeCaptioner:
    def __init__(self):
        self.calls: list[tuple[Path, str | None, int]] = []

    def caption(self, image_path: Path, prompt: str | None = None, max_new_tokens: int = 64) -> str:
        self.calls.append((image_path, prompt, max_new_tokens))
        return f'"caption for {image_path.name}"'


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


def test_clean_caption_strips_quotes_and_truncates_on_word_boundary():
    assert clean_caption(' "A person opens a box." ', max_chars=100) == "A person opens a box."
    assert clean_caption("alpha beta gamma delta", max_chars=14) == "alpha beta..."


def test_caption_cache_round_trips_latest_record(tmp_path):
    path = tmp_path / "captions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        append_caption_record(handle, CaptionRecord("sample-1", 1, "a.jpg", "old"))
        append_caption_record(handle, CaptionRecord("sample-1", 1, "a.jpg", "new"))

    cache = load_caption_cache(path)
    assert cache[("sample-1", 1, "a.jpg")] == "new"


def test_captions_for_row_can_fail_or_fill_empty_for_missing_values():
    row = make_row()
    with pytest.raises(ValueError, match="Missing cached captions"):
        captions_for_row(row, {}, missing_policy="fail")
    assert captions_for_row(row, {}, missing_policy="empty") == ["", "", "", ""]


def test_generate_captions_for_row_uses_cache_and_appends_missing_records(tmp_path):
    row = make_row()
    cache = {("sample-1", 1, "a.jpg"): "cached first"}
    captioner = FakeCaptioner()
    path = tmp_path / "captions.jsonl"

    with path.open("w", encoding="utf-8") as handle:
        captions = generate_captions_for_row(
            row=row,
            image_dir=Path("/data/test"),
            captioner=captioner,
            cache=cache,
            cache_handle=handle,
            caption_max_new_tokens=12,
            sentence_aware=True,
        )

    assert captions == [
        "cached first",
        "caption for b.jpg",
        "caption for c.jpg",
        "caption for d.jpg",
    ]
    assert len(captioner.calls) == 3
    assert all(call[1] is not None for call in captioner.calls)
    assert load_caption_cache(path)[("sample-1", 4, "d.jpg")] == "caption for d.jpg"


def test_default_caption_cache_path_is_namespaced_by_idea():
    assert default_caption_cache_path("test") == Path("outputs/caption_augmented/test_captions.jsonl")
