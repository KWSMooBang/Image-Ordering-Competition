"""Dataset for caption-aware pairwise comparison rows."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .captions import compose_pair_text


REQUIRED_PAIR_COLUMNS = {
    "sample_id",
    "pair_id",
    "sentence",
    "image_a_index",
    "image_b_index",
    "image_a_path",
    "image_b_path",
    "label",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "y", "t"}:
        return True
    if normalised in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


class PairwiseDataset(Dataset):
    """Load two images, the sentence/captions text, and a binary order label.

    Label semantics:
        1.0 -> image A occurs before image B
        0.0 -> image B occurs before image A
    """

    def __init__(
        self,
        pairs: str | Path | pd.DataFrame,
        image_root: str | Path,
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        pair_transform: Callable[[Image.Image, Image.Image], tuple[Any, Any]] | None = None,
        swap_probability: float = 0.0,
        no_ordering_filter: bool | None = None,
        return_paths: bool = False,
        strict_images: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(pairs, pd.DataFrame):
            data = pairs.copy()
        else:
            pair_path = Path(pairs)
            if not pair_path.exists():
                raise FileNotFoundError(f"Pair CSV not found: {pair_path}")
            data = pd.read_csv(pair_path)

        missing_columns = sorted(REQUIRED_PAIR_COLUMNS - set(data.columns))
        if missing_columns:
            raise ValueError(f"Pair data is missing columns: {missing_columns}. Available columns: {list(data.columns)}")

        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability must be between 0 and 1.")

        labels = pd.to_numeric(data["label"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            bad_rows = data.loc[labels.isna() | ~labels.isin([0, 1])].head(5)
            raise ValueError(f"Pair labels must be 0 or 1. Invalid rows:\n{bad_rows}")
        data["label"] = labels.astype(int)

        if "image_a_caption" not in data.columns:
            data["image_a_caption"] = ""
        if "image_b_caption" not in data.columns:
            data["image_b_caption"] = ""

        if no_ordering_filter is not None:
            if "no_ordering" not in data.columns:
                raise ValueError("no_ordering_filter was provided, but pair data has no 'no_ordering' column.")
            mask = data["no_ordering"].map(_as_bool) == no_ordering_filter
            data = data.loc[mask].copy()

        self.data = data.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.pair_transform = pair_transform
        self.swap_probability = float(swap_probability)
        self.return_paths = return_paths
        self.strict_images = strict_images

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, relative_or_absolute: object) -> Path:
        path = Path(str(relative_or_absolute))
        return path if path.is_absolute() else self.image_root / path

    def _load_rgb(self, path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except FileNotFoundError:
            if self.strict_images:
                raise FileNotFoundError(f"Image not found: {path}") from None
            raise
        except (UnidentifiedImageError, OSError) as exc:
            if self.strict_images:
                raise RuntimeError(f"Could not read image: {path}") from exc
            raise

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]

        image_a_path = self._resolve_path(row["image_a_path"])
        image_b_path = self._resolve_path(row["image_b_path"])
        image_a = self._load_rgb(image_a_path)
        image_b = self._load_rgb(image_b_path)

        image_a_index = int(row["image_a_index"])
        image_b_index = int(row["image_b_index"])
        image_a_caption = str(row.get("image_a_caption", ""))
        image_b_caption = str(row.get("image_b_caption", ""))
        label = float(row["label"])

        swapped = self.swap_probability > 0 and random.random() < self.swap_probability
        if swapped:
            image_a, image_b = image_b, image_a
            image_a_path, image_b_path = image_b_path, image_a_path
            image_a_index, image_b_index = image_b_index, image_a_index
            image_a_caption, image_b_caption = image_b_caption, image_a_caption
            label = 1.0 - label

        if self.pair_transform is not None:
            image_a, image_b = self.pair_transform(image_a, image_b)
        elif self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        sample: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "pair_id": str(row["pair_id"]),
            "sentence": str(row["sentence"]),
            "text": compose_pair_text(row["sentence"], image_a_index, image_b_index, image_a_caption, image_b_caption),
            "image_a": image_a,
            "image_b": image_b,
            "image_a_index": torch.tensor(image_a_index, dtype=torch.long),
            "image_b_index": torch.tensor(image_b_index, dtype=torch.long),
            "image_a_caption": image_a_caption,
            "image_b_caption": image_b_caption,
            "label": torch.tensor(label, dtype=torch.float32),
            "no_ordering": _as_bool(row.get("no_ordering", False)),
            "swapped": swapped,
        }

        if self.return_paths:
            sample["image_a_path"] = str(image_a_path)
            sample["image_b_path"] = str(image_b_path)
        return sample

    def class_counts(self) -> dict[int, int]:
        counts = self.data["label"].value_counts().to_dict()
        return {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
