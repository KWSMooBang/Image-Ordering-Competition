from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

from src.caption_augmented_common import get_or_generate_captions, load_caption_cache
from src.data_utils import read_csv
from src.qwen_vl_common import DEFAULT_MODEL_NAME, load_qwen_vl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen VLM image captions into a JSONL cache.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--caption-max-new-tokens", type=int, default=80)
    parser.add_argument("--max-caption-chars", type=int, default=360)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional attention backend passed to from_pretrained.",
    )
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--refresh-captions", action="store_true")
    return parser.parse_args()


def default_cache_path(split: str) -> Path:
    return Path("outputs") / f"{split}_caption_augmented_qwen_captions.jsonl"


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_df = read_csv(data_dir / f"{args.split}.csv")
    if args.max_samples is not None:
        split_df = split_df.head(args.max_samples).copy()

    caption_cache_path = Path(args.caption_cache) if args.caption_cache else default_cache_path(args.split)
    caption_cache_path.parent.mkdir(parents=True, exist_ok=True)
    caption_cache = load_caption_cache(caption_cache_path)

    print(f"Loading caption model: {args.model_name}")
    bundle = load_qwen_vl(
        args.model_name,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    image_dir = data_dir / args.split
    print(f"Generating captions for {len(split_df)} {args.split} samples")
    with caption_cache_path.open("a", encoding="utf-8") as caption_handle:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            get_or_generate_captions(
                row=row,
                image_dir=image_dir,
                bundle=bundle,
                cache=caption_cache,
                cache_handle=caption_handle,
                caption_max_new_tokens=args.caption_max_new_tokens,
                max_caption_chars=args.max_caption_chars,
                refresh_captions=args.refresh_captions,
            )

    print(f"Saved caption cache to {caption_cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
