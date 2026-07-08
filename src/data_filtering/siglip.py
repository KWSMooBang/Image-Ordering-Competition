from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src.data_filtering.quality import FrameRelevanceScores

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
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP processor loading failed because tokenizer dependencies are missing. "
                "Install `sentencepiece` and `protobuf` in the runtime environment, then restart "
                "the Python process."
            ) from exc
        model_kwargs: dict[str, Any] = {}
        dtype = _resolve_torch_dtype(torch, torch_dtype, self.device)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()

    def __call__(
        self,
        row: Mapping[str, Any],
        image_paths: Sequence[Path],
        captions: Sequence[str | None],
    ) -> FrameRelevanceScores:
        sentence = str(row.get("Sentence", ""))
        image_scores: list[float | None] = [None] * len(image_paths)
        caption_scores: list[float | None] = []
        if len(captions) != len(image_paths):
            raise ValueError(f"Expected {len(image_paths)} captions, got {len(captions)}")
        if not sentence.strip():
            return FrameRelevanceScores(
                relevance_scores=[None] * len(image_paths),
                image_relevance_scores=image_scores,
                caption_embedding_scores=caption_scores,
            )

        valid_items = [(index, path) for index, path in enumerate(image_paths) if path.exists()]
        for start in range(0, len(valid_items), self.batch_size):
            batch_items = valid_items[start : start + self.batch_size]
            batch_paths = [path for _, path in batch_items]
            batch_scores = self._score_batch(sentence, batch_paths)
            for (index, _), score in zip(batch_items, batch_scores, strict=True):
                image_scores[index] = score

        if any(caption and caption.strip() for caption in captions):
            caption_scores = self._score_captions(sentence, captions)
        relevance_scores = _combine_scores(image_scores, caption_scores)
        return FrameRelevanceScores(
            relevance_scores=relevance_scores,
            image_relevance_scores=image_scores,
            caption_embedding_scores=caption_scores,
        )

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

    def _score_captions(self, sentence: str, captions: Sequence[str | None]) -> list[float | None]:
        scores: list[float | None] = [None] * len(captions)
        valid_items = [
            (index, str(caption).strip())
            for index, caption in enumerate(captions)
            if caption is not None and str(caption).strip()
        ]
        if not valid_items:
            return scores

        sentence_features = self._encode_texts([sentence])
        for start in range(0, len(valid_items), self.batch_size):
            batch_items = valid_items[start : start + self.batch_size]
            batch_texts = [caption for _, caption in batch_items]
            caption_features = self._encode_texts(batch_texts)
            similarities = caption_features @ sentence_features.T
            for (index, _), score in zip(batch_items, similarities[:, 0].detach().cpu().tolist(), strict=True):
                scores[index] = float(score)
        return scores

    def _encode_texts(self, texts: Sequence[str]):
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return _normalize(_pooled_tensor(features).float())


def _combine_scores(
    image_scores: Sequence[float | None],
    caption_scores: Sequence[float | None],
) -> list[float | None]:
    combined: list[float | None] = []
    for index, image_score in enumerate(image_scores):
        candidates = [image_score]
        if caption_scores:
            candidates.append(caption_scores[index])
        valid_scores = [score for score in candidates if score is not None]
        combined.append(min(valid_scores) if valid_scores else None)
    return combined


def _normalize(tensor):
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def _pooled_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if isinstance(output, tuple):
        if len(output) > 1 and output[1] is not None:
            return output[1]
        if output:
            return output[0]
    return output


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
