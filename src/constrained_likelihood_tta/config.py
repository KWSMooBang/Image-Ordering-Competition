from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
DEFAULT_ORDER_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT_DIR = "outputs/constrained_likelihood_tta"


@dataclass(frozen=True)
class Defaults:
    caption_model: str = DEFAULT_CAPTION_MODEL
    order_model: str = DEFAULT_ORDER_MODEL
    train_caption_cache: str = f"{DEFAULT_OUTPUT_DIR}/train_captions.jsonl"
    test_caption_cache: str = f"{DEFAULT_OUTPUT_DIR}/test_captions.jsonl"
    output: str = f"{DEFAULT_OUTPUT_DIR}/submission.csv"
    raw_output: str = f"{DEFAULT_OUTPUT_DIR}/raw_outputs.jsonl"
    caption_max_new_tokens: int = 64
    max_caption_chars: int = 320
    candidate_batch_size: int = 4
    tta_permutations: int = 4
