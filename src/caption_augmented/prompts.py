from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS


def build_caption_prompt(row: pd.Series, image_index: int) -> str:
    return (
        f'Story sentence: "{row["Sentence"]}"\n'
        f"This is Image {image_index} from a shuffled four-frame story. "
        "Write one concise caption for only this image. Focus on visible actions, "
        "object states, positions, and before/after clues that could help decide chronology. "
        "Do not guess the final image order. Return one sentence only."
    )


def build_order_messages(row: pd.Series, image_dir: Path, captions: list[str]) -> list[dict[str, Any]]:
    if len(captions) != 4:
        raise ValueError(f"Expected 4 captions, got {len(captions)}")

    content: list[dict[str, str]] = []
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        image_path = image_dir / str(row["Id"]) / str(row[column])
        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": f"\nImage {image_index} caption: {captions[image_index - 1]}\n"})

    content.append(
        {
            "type": "text",
            "text": (
                f'Story sentence: "{row["Sentence"]}"\n'
                "The captions above were generated automatically and may be imperfect. "
                "Use the images as primary evidence and the captions as supporting notes. "
                "Determine the correct chronological order of Image 1 to Image 4 to match the sentence. "
                "Return ONLY a Python list of chronological image labels. Example: [1, 2, 3, 4]"
            ),
        }
    )
    return [{"role": "user", "content": content}]
