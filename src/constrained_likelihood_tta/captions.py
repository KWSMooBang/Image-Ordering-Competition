from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd
from tqdm.auto import tqdm

from src.constrained_likelihood_tta.config import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_ORDER_MODEL,
    Defaults,
)
from src.constrained_likelihood_tta.model import BlipCaptioner, Captioner, QwenCaptioner
from src.constrained_likelihood_tta.prompts import build_caption_prompt
from src.data_utils import INPUT_COLUMNS, image_paths_for_row, read_csv


@dataclass(frozen=True)
class CaptionRecord:
    Id: object
    image_index: int
    image: str
    caption: str


CaptionCache = dict[tuple[str, int, str], str]


def cache_key(
    row_id: object, image_index: int, image_name: object
) -> tuple[str, int, str]:
    return (str(row_id), int(image_index), str(image_name))


def clean_caption(text: str, *, max_chars: int) -> str:
    value = " ".join(str(text).strip().split())
    if len(value) <= max_chars:
        return value
    shortened = value[: max_chars + 1].rsplit(" ", maxsplit=1)[0].rstrip()
    return f"{shortened}..."


def load_caption_cache(path: str | Path | None) -> CaptionCache:
    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    cache: CaptionCache = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[cache_key(record["Id"], record["image_index"], record["image"])] = (
                str(record["caption"])
            )
    return cache


def get_cached_caption(
    cache: CaptionCache,
    row_id: object,
    image_index: int,
    image_name: object,
) -> str | None:
    exact = cache.get(cache_key(row_id, image_index, image_name))
    if exact is not None:
        return exact
    normalized_id, normalized_image = str(row_id), str(image_name)
    for (cache_id, _cache_index, cache_image), caption in cache.items():
        if cache_id == normalized_id and cache_image == normalized_image:
            return caption
    return None


def captions_for_row(
    row: pd.Series,
    cache: CaptionCache,
    *,
    missing_policy: str,
) -> list[str]:
    values: list[str] = []
    missing: list[str] = []
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        image_name = str(row[column])
        caption = get_cached_caption(cache, row["Id"], image_index, image_name)
        if caption is None:
            missing.append(f"Id={row['Id']} image={image_name}")
            caption = ""
        values.append(caption)
    if missing and missing_policy == "fail":
        raise ValueError("Missing captions, e.g. " + "; ".join(missing[:5]))
    return values


def append_record(handle: TextIO, record: CaptionRecord) -> None:
    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    handle.flush()


def generate_captions_for_row(
    row: pd.Series,
    image_dir: Path,
    captioner: Captioner,
    cache: CaptionCache,
    handle: TextIO,
    *,
    refresh: bool,
    sentence_aware: bool,
    max_new_tokens: int,
    max_chars: int,
) -> list[str]:
    captions: list[str] = []
    for image_index, image_path in enumerate(
        image_paths_for_row(row, image_dir), start=1
    ):
        image_name = str(row[INPUT_COLUMNS[image_index - 1]])
        key = cache_key(row["Id"], image_index, image_name)
        caption = (
            None
            if refresh
            else get_cached_caption(
                cache,
                row["Id"],
                image_index,
                image_name,
            )
        )
        if caption is None:
            prompt = build_caption_prompt(row, image_index) if sentence_aware else None
            caption = clean_caption(
                captioner.caption(image_path, prompt, max_new_tokens),
                max_chars=max_chars,
            )
            cache[key] = caption
            append_record(
                handle, CaptionRecord(row["Id"], image_index, image_name, caption)
            )
        captions.append(caption)
    return captions


def parse_args() -> argparse.Namespace:
    defaults = Defaults()
    parser = argparse.ArgumentParser(
        description="Generate captions for constrained likelihood TTA."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--caption-backend", choices=["blip", "qwen"], default="blip")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--qwen-caption-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument(
        "--caption-device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--caption-torch-dtype", choices=["auto", "float16", "float32"], default="auto"
    )
    parser.add_argument(
        "--qwen-torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument(
        "--caption-max-new-tokens", type=int, default=defaults.caption_max_new_tokens
    )
    parser.add_argument(
        "--max-caption-chars", type=int, default=defaults.max_caption_chars
    )
    parser.add_argument("--sentence-aware", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = Defaults()
    data_dir = Path(args.data_dir)
    dataframe = read_csv(data_dir / f"{args.split}.csv")
    if args.max_samples is not None:
        dataframe = dataframe.head(args.max_samples).copy()
    default_output = (
        defaults.train_caption_cache
        if args.split == "train"
        else defaults.test_caption_cache
    )
    output_path = Path(args.output or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_caption_cache(output_path)

    if args.caption_backend == "blip":
        captioner: Captioner = BlipCaptioner(
            args.caption_model,
            device=args.caption_device,
            torch_dtype=args.caption_torch_dtype,
        )
    else:
        captioner = QwenCaptioner(
            args.qwen_caption_model,
            device_map=args.device_map,
            torch_dtype=args.qwen_torch_dtype,
            attn_implementation=args.attn_implementation,
        )

    with output_path.open("a", encoding="utf-8") as handle:
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            generate_captions_for_row(
                row,
                data_dir / args.split,
                captioner,
                cache,
                handle,
                refresh=args.refresh,
                sentence_aware=args.sentence_aware,
                max_new_tokens=args.caption_max_new_tokens,
                max_chars=args.max_caption_chars,
            )
    print(f"Saved captions to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
