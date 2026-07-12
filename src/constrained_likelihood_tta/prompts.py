from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_utils import INPUT_COLUMNS


def build_caption_prompt(row: pd.Series, image_index: int) -> str:
    return (
        f'Story sentence: "{row["Sentence"]}"\n'
        f"This is Image {image_index} from a shuffled four-frame story. "
        "Describe only this image in one concise sentence. Focus on visible actions, "
        "object states, positions, camera motion clues, and before/after evidence. "
        "Do not guess the final image order."
    )


def build_order_messages(
    row: pd.Series,
    image_dir: Path,
    captions: list[str],
) -> list[dict[str, Any]]:
    if len(captions) != len(INPUT_COLUMNS):
        raise ValueError(f"Expected {len(INPUT_COLUMNS)} captions, got {len(captions)}")

    content: list[dict[str, str]] = []
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        image_path = image_dir / str(row["Id"]) / str(row[column])
        content.append({"type": "image", "image": str(image_path)})
        content.append(
            {
                "type": "text",
                "text": f"\nImage {image_index} caption: {captions[image_index - 1]}\n",
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                f'Story sentence: "{row["Sentence"]}"\n'
                "The captions may be imperfect; use the images as primary evidence. "
                "Determine the chronological order of Image 1 through Image 4. "
                "Return only a Python list such as [1, 2, 3, 4]."
            ),
        }
    )
    return [{"role": "user", "content": content}]
