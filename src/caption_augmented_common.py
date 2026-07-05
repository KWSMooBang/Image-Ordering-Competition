from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import pandas as pd

from src.data_utils import INPUT_COLUMNS, image_paths_for_row
from src.qwen_vl_common import LoadedVLModel, generate_text


def get_caption_message(row: pd.Series, image_path: Path, image_index: int) -> list[dict[str, object]]:
    sentence = row["Sentence"]
    prompt = (
        f'Story sentence: "{sentence}"\n'
        f"This is Image {image_index} from a shuffled four-frame story. "
        "Write one concise English caption for only this image. Focus on visible actions, "
        "object states, positions, and before/after clues that could help decide the chronology. "
        "Do not guess the final image order. Return one sentence only."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def get_order_message(row: pd.Series, image_dir: Path, captions: list[str]) -> list[dict[str, object]]:
    content: list[dict[str, str]] = []
    for index, column in enumerate(INPUT_COLUMNS, start=1):
        image_path = image_dir / str(row["Id"]) / str(row[column])
        caption = captions[index - 1]
        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": f"\nImage {index} caption: {caption}\n"})

    sentence = row["Sentence"]
    user_text = (
        f'Story sentence: "{sentence}"\n'
        "The captions above were generated automatically and may be imperfect, so use the images "
        "as the primary evidence and the captions as supporting notes. "
        "Determine the correct chronological order of Image 1 to Image 4 to match the sentence. "
        "Provide the answer ONLY as a Python list of the chronological image labels. "
        "Example: [1, 2, 3, 4]"
    )
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def clean_caption(text: str, max_chars: int) -> str:
    caption = " ".join(text.strip().split())
    if (caption.startswith('"') and caption.endswith('"')) or (caption.startswith("'") and caption.endswith("'")):
        caption = caption[1:-1].strip()
    if len(caption) <= max_chars:
        return caption
    truncated = caption[: max_chars + 1].rsplit(" ", maxsplit=1)[0].rstrip()
    return f"{truncated}..."


def caption_cache_key(row_id: object, image_index: int, image_name: object) -> tuple[str, int, str]:
    return (str(row_id), int(image_index), str(image_name))


def load_caption_cache(path: str | Path) -> dict[tuple[str, int, str], str]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    cache: dict[tuple[str, int, str], str] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = caption_cache_key(record["Id"], record["image_index"], record["image"])
            cache[key] = str(record["caption"])
    return cache


def append_caption_cache(
    handle: TextIO,
    row_id: object,
    image_index: int,
    image_name: object,
    caption: str,
) -> None:
    handle.write(
        json.dumps(
            {
                "Id": row_id,
                "image_index": image_index,
                "image": image_name,
                "caption": caption,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


def get_or_generate_captions(
    row: pd.Series,
    image_dir: Path,
    bundle: LoadedVLModel,
    cache: dict[tuple[str, int, str], str],
    cache_handle: TextIO,
    caption_max_new_tokens: int,
    max_caption_chars: int,
    refresh_captions: bool,
) -> list[str]:
    captions: list[str] = []
    paths = image_paths_for_row(row, image_dir)
    for image_index, image_path in enumerate(paths, start=1):
        image_name = row[INPUT_COLUMNS[image_index - 1]]
        key = caption_cache_key(row["Id"], image_index, image_name)
        caption = None if refresh_captions else cache.get(key)
        if caption is None:
            messages = get_caption_message(row, image_path, image_index)
            raw_caption = generate_text(bundle, messages, max_new_tokens=caption_max_new_tokens)
            caption = clean_caption(raw_caption, max_chars=max_caption_chars)
            cache[key] = caption
            append_caption_cache(cache_handle, row["Id"], image_index, image_name, caption)
        captions.append(caption)
    return captions
