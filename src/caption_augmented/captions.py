from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd
from tqdm.auto import tqdm

from src.caption_augmented.config import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_ORDER_MODEL,
    CaptionAugmentedDefaults,
)
from src.caption_augmented.model import BlipCaptioner, Captioner, QwenCaptioner
from src.caption_augmented.prompts import build_caption_prompt
from src.data_utils import INPUT_COLUMNS, image_paths_for_row, read_csv


@dataclass(frozen=True)
class CaptionRecord:
    Id: object
    image_index: int
    image: str
    caption: str


CaptionCache = dict[tuple[str, int, str], str]


def caption_cache_key(row_id: object, image_index: int, image_name: object) -> tuple[str, int, str]:
    return (str(row_id), int(image_index), str(image_name))


def clean_caption(text: str, max_chars: int = CaptionAugmentedDefaults.max_caption_chars) -> str:
    caption = " ".join(str(text).strip().split())
    if (caption.startswith('"') and caption.endswith('"')) or (caption.startswith("'") and caption.endswith("'")):
        caption = caption[1:-1].strip()
    if len(caption) <= max_chars:
        return caption
    truncated = caption[: max_chars + 1].rsplit(" ", maxsplit=1)[0].rstrip()
    return f"{truncated}..."


def load_caption_cache(path: str | Path) -> CaptionCache:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    cache: CaptionCache = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = caption_cache_key(record["Id"], record["image_index"], record["image"])
            cache[key] = str(record["caption"])
    return cache


def get_cached_caption(
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


def append_caption_record(handle: TextIO, record: CaptionRecord) -> None:
    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    handle.flush()


def captions_for_row(
    row: pd.Series,
    cache: CaptionCache,
    *,
    missing_policy: str = "empty",
) -> list[str]:
    captions: list[str] = []
    missing: list[str] = []
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        image_name = str(row[column])
        caption = get_cached_caption(cache, row["Id"], image_index, image_name)
        if caption is None:
            missing.append(f"Id={row['Id']} image_index={image_index} image={image_name}")
            caption = ""
        captions.append(caption)

    if missing and missing_policy == "fail":
        raise ValueError("Missing cached captions, e.g. " + "; ".join(missing[:5]))
    return captions


def generate_captions_for_row(
    row: pd.Series,
    image_dir: Path,
    captioner: Captioner,
    cache: CaptionCache,
    cache_handle: TextIO,
    *,
    refresh: bool = False,
    caption_max_new_tokens: int = CaptionAugmentedDefaults.caption_max_new_tokens,
    max_caption_chars: int = CaptionAugmentedDefaults.max_caption_chars,
    sentence_aware: bool = False,
) -> list[str]:
    captions: list[str] = []
    image_paths = image_paths_for_row(row, image_dir)
    for image_index, image_path in enumerate(image_paths, start=1):
        image_name = str(row[INPUT_COLUMNS[image_index - 1]])
        key = caption_cache_key(row["Id"], image_index, image_name)
        caption = None if refresh else get_cached_caption(cache, row["Id"], image_index, image_name)
        if caption is None:
            prompt = build_caption_prompt(row, image_index) if sentence_aware else None
            raw_caption = captioner.caption(image_path, prompt=prompt, max_new_tokens=caption_max_new_tokens)
            caption = clean_caption(raw_caption, max_chars=max_caption_chars)
            cache[key] = caption
            append_caption_record(
                cache_handle,
                CaptionRecord(
                    Id=row["Id"],
                    image_index=image_index,
                    image=image_name,
                    caption=caption,
                ),
            )
        captions.append(caption)
    return captions


def generate_fresh_captions_for_row(
    row: pd.Series,
    image_dir: Path,
    captioner: Captioner,
    *,
    caption_max_new_tokens: int = CaptionAugmentedDefaults.caption_max_new_tokens,
    max_caption_chars: int = CaptionAugmentedDefaults.max_caption_chars,
    sentence_aware: bool = False,
) -> list[str]:
    captions: list[str] = []
    image_paths = image_paths_for_row(row, image_dir)
    for image_index, image_path in enumerate(image_paths, start=1):
        prompt = build_caption_prompt(row, image_index) if sentence_aware else None
        raw_caption = captioner.caption(image_path, prompt=prompt, max_new_tokens=caption_max_new_tokens)
        captions.append(clean_caption(raw_caption, max_chars=max_caption_chars))
    return captions


def default_caption_cache_path(split: str) -> Path:
    return Path("outputs") / "caption_augmented" / f"{split}_captions.jsonl"


def build_captioner(args: argparse.Namespace) -> Captioner:
    if args.caption_backend == "blip":
        return BlipCaptioner(
            model_name=args.caption_model,
            device=args.caption_device,
            torch_dtype=args.caption_torch_dtype,
        )
    if args.caption_backend == "qwen":
        return QwenCaptioner(
            model_name=args.qwen_caption_model,
            device_map=args.device_map,
            torch_dtype=args.qwen_torch_dtype,
            attn_implementation=args.attn_implementation,
        )
    raise ValueError(f"Unsupported caption backend: {args.caption_backend}")


def parse_args() -> argparse.Namespace:
    defaults = CaptionAugmentedDefaults()
    parser = argparse.ArgumentParser(description="Generate caption cache for the caption-augmented idea.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--caption-backend", choices=["blip", "qwen"], default="blip")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--qwen-caption-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--caption-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--caption-torch-dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--qwen-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--caption-max-new-tokens", type=int, default=defaults.caption_max_new_tokens)
    parser.add_argument("--max-caption-chars", type=int, default=defaults.max_caption_chars)
    parser.add_argument("--sentence-aware", action="store_true", help="Pass sentence-aware prompts to the caption model")
    parser.add_argument("--refresh-captions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_df = read_csv(data_dir / f"{args.split}.csv")
    if args.max_samples is not None:
        split_df = split_df.head(args.max_samples).copy()

    output_path = Path(args.output) if args.output else default_caption_cache_path(args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_caption_cache(output_path)

    print(f"Loading caption backend: {args.caption_backend}")
    captioner = build_captioner(args)
    image_dir = data_dir / args.split

    print(f"Generating captions for {len(split_df)} {args.split} samples")
    with output_path.open("a", encoding="utf-8") as handle:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            generate_captions_for_row(
                row=row,
                image_dir=image_dir,
                captioner=captioner,
                cache=cache,
                cache_handle=handle,
                refresh=args.refresh_captions,
                caption_max_new_tokens=args.caption_max_new_tokens,
                max_caption_chars=args.max_caption_chars,
                sentence_aware=args.sentence_aware,
            )

    print(f"Saved caption cache to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
