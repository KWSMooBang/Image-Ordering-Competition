"""Caption helpers for pairwise ordering.

Caption records are compatible with the caption-augmented JSONL/CSV format:
``Id``, ``image_index``, ``image``, and ``caption``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


CaptionCache = dict[tuple[str, int, str], str]


def caption_cache_key(row_id: object, image_index: int, image_name: object) -> tuple[str, int, str]:
    return (str(row_id), int(image_index), str(image_name))


def load_caption_cache(path: str | Path | None) -> CaptionCache:
    if path is None:
        return {}

    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    if cache_path.suffix.lower() == ".csv":
        return _load_caption_cache_csv(cache_path)
    return _load_caption_cache_jsonl(cache_path)


def _load_caption_cache_jsonl(path: Path) -> CaptionCache:
    cache: CaptionCache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = caption_cache_key(record["Id"], record["image_index"], record["image"])
            cache[key] = str(record["caption"])
    return cache


def _load_caption_cache_csv(path: Path) -> CaptionCache:
    df = pd.read_csv(path)
    required = {"Id", "image_index", "image", "caption"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Caption CSV is missing columns: {missing}")

    cache: CaptionCache = {}
    for record in df.itertuples(index=False):
        key = caption_cache_key(record.Id, int(record.image_index), record.image)
        cache[key] = str(record.caption)
    return cache


def lookup_caption(
    cache: CaptionCache,
    row_id: object,
    image_index: int,
    image_name: object,
) -> str | None:
    exact_key = caption_cache_key(row_id, image_index, image_name)
    caption = cache.get(exact_key)
    if caption is not None:
        return caption

    normalized_id = str(row_id)
    normalized_image = str(image_name)
    for cache_id, cache_image_index, cache_image in cache:
        if cache_id == normalized_id and cache_image == normalized_image:
            return cache[(cache_id, cache_image_index, cache_image)]
    return None


def captions_for_image_files(
    *,
    row_id: object,
    image_files: Mapping[int, str],
    caption_cache: CaptionCache,
    missing_policy: str = "empty",
) -> dict[int, str]:
    captions: dict[int, str] = {}
    missing: list[str] = []
    for image_index, image_name in image_files.items():
        caption = lookup_caption(caption_cache, row_id, image_index, image_name)
        if caption is None:
            missing.append(f"Id={row_id} image_index={image_index} image={image_name}")
            caption = ""
        captions[int(image_index)] = caption

    if missing and missing_policy == "fail":
        raise ValueError("Missing cached captions, e.g. " + "; ".join(missing[:5]))
    if missing_policy not in {"empty", "fail"}:
        raise ValueError("--caption-missing-policy must be one of: empty, fail")
    return captions


def clean_text_part(value: object) -> str:
    return " ".join(str(value).strip().split())


def compose_pair_text(
    sentence: object,
    image_a_index: int,
    image_b_index: int,
    image_a_caption: object | None = None,
    image_b_caption: object | None = None,
) -> str:
    parts = [
        f'Story sentence: "{clean_text_part(sentence)}"',
        f"Image A is original Image {int(image_a_index)}.",
        f"Image B is original Image {int(image_b_index)}.",
    ]
    caption_a = clean_text_part(image_a_caption or "")
    caption_b = clean_text_part(image_b_caption or "")
    if caption_a:
        parts.append(f"Image A caption: {caption_a}")
    if caption_b:
        parts.append(f"Image B caption: {caption_b}")
    parts.append("Predict whether Image A happens before Image B in the story.")
    return "\n".join(parts)


def caption_columns_present(columns: Sequence[str]) -> bool:
    return {"image_a_caption", "image_b_caption"}.issubset(set(columns))
