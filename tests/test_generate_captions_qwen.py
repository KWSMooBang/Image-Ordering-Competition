from pathlib import Path

from src.generate_captions_qwen import default_cache_path


def test_default_cache_path_uses_outputs_for_split():
    assert default_cache_path("train") == Path("outputs/train_caption_augmented_qwen_captions.jsonl")
    assert default_cache_path("test") == Path("outputs/test_caption_augmented_qwen_captions.jsonl")
