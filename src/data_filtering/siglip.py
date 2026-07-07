from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_SIGLIP_MODEL = "google/siglip-so400m-patch14-384"


class SiglipImageTextScorer:
    """Score each image frame against the row sentence with SigLIP sigmoid logits."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        batch_size: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("SigLIP relevance scoring requires torch and transformers.") from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        dtype = _resolve_torch_dtype(torch, torch_dtype, self.device)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()

    def __call__(self, row: Mapping[str, Any], image_paths: Sequence[Path]) -> list[float | None]:
        sentence = str(row.get("Sentence", ""))
        scores: list[float | None] = [None] * len(image_paths)
        valid_items = [(index, path) for index, path in enumerate(image_paths) if path.exists()]
        if not sentence.strip() or not valid_items:
            return scores

        for start in range(0, len(valid_items), self.batch_size):
            batch_items = valid_items[start : start + self.batch_size]
            batch_paths = [path for _, path in batch_items]
            batch_scores = self._score_batch(sentence, batch_paths)
            for (index, _), score in zip(batch_items, batch_scores, strict=True):
                scores[index] = score
        return scores

    def _score_batch(self, sentence: str, image_paths: Sequence[Path]) -> list[float]:
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        inputs = self.processor(
            text=[sentence],
            images=images,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = self.torch.sigmoid(outputs.logits_per_image)
        return [float(value) for value in probabilities[:, 0].detach().cpu().tolist()]


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(torch, dtype_name: str, device: str):
    if dtype_name == "auto":
        if device == "cuda":
            return torch.float16
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
