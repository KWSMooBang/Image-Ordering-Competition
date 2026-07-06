from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
DEFAULT_ORDER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CAPTION_CACHE = "outputs/caption_augmented/test_captions.jsonl"
DEFAULT_OUTPUT = "outputs/caption_augmented/submission.csv"
DEFAULT_RAW_OUTPUT = "outputs/caption_augmented/raw_outputs.jsonl"


@dataclass(frozen=True)
class CaptionAugmentedDefaults:
    caption_model: str = DEFAULT_CAPTION_MODEL
    order_model: str = DEFAULT_ORDER_MODEL
    caption_cache: str = DEFAULT_CAPTION_CACHE
    output: str = DEFAULT_OUTPUT
    raw_output: str = DEFAULT_RAW_OUTPUT
    caption_max_new_tokens: int = 64
    order_max_new_tokens: int = 128
    max_caption_chars: int = 320
