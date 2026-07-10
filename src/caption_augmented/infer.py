from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from src.caption_augmented.captions import (
    build_captioner,
    generate_fresh_captions_for_row,
)
from src.caption_augmented.config import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_ORDER_MODEL,
    CaptionAugmentedDefaults,
)
from src.caption_augmented.model import QwenOrderer
from src.caption_augmented.prompts import build_order_messages
from src.data_utils import read_csv
from src.submission import format_answer, normalize_permutation, parse_model_output, write_submission


def parse_args() -> argparse.Namespace:
    defaults = CaptionAugmentedDefaults()
    parser = argparse.ArgumentParser(description="Run caption-augmented Qwen ordering inference.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=defaults.output)
    parser.add_argument("--raw-output", default=defaults.raw_output)
    parser.add_argument("--caption-cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fallback-answer", default="[1, 2, 3, 4]")

    parser.add_argument(
        "--caption-missing-policy",
        choices=["generate", "fail", "empty"],
        default="generate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--caption-backend", choices=["blip", "qwen"], default="blip")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--qwen-caption-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--caption-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--caption-torch-dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--caption-max-new-tokens", type=int, default=defaults.caption_max_new_tokens)
    parser.add_argument("--max-caption-chars", type=int, default=defaults.max_caption_chars)
    parser.add_argument("--sentence-aware-captions", action="store_true")
    parser.add_argument("--refresh-captions", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--order-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--order-adapter", default=None, help="Optional LoRA/PEFT adapter directory from training")
    parser.add_argument("--order-max-new-tokens", type=int, default=defaults.order_max_new_tokens)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--qwen-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def resolve_captions_for_row(row, image_dir: Path, args: argparse.Namespace, captioner):
    return generate_fresh_captions_for_row(
        row=row,
        image_dir=image_dir,
        captioner=captioner,
        caption_max_new_tokens=args.caption_max_new_tokens,
        max_caption_chars=args.max_caption_chars,
        sentence_aware=args.sentence_aware_captions,
    )


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    test_df = read_csv(data_dir / "test.csv")
    sample_df = read_csv(data_dir / "sample_submission.csv")
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples).copy()
        sample_df = sample_df.head(args.max_samples).copy()

    fallback = normalize_permutation(ast.literal_eval(args.fallback_answer))
    image_dir = data_dir / "test"

    if args.caption_cache:
        print("Ignoring --caption-cache during inference; captions are generated fresh for every sample.")
    print(f"Loading caption backend: {args.caption_backend}")
    captioner = build_captioner(args)

    print(f"Loading ordering model: {args.order_model}")
    orderer = QwenOrderer(
        model_name=args.order_model,
        device_map=args.device_map,
        torch_dtype=args.qwen_torch_dtype,
        attn_implementation=args.attn_implementation,
        adapter_path=args.order_adapter,
    )

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, str]] = []
    print(f"Running fresh-caption inference on {len(test_df)} samples")
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            captions = resolve_captions_for_row(
                row=row,
                image_dir=image_dir,
                args=args,
                captioner=captioner,
            )
            messages = build_order_messages(row, image_dir=image_dir, captions=captions)
            output_text = orderer.generate_order(messages, max_new_tokens=args.order_max_new_tokens)

            try:
                pred_list = parse_model_output(output_text, fallback=fallback)
            except ValueError:
                pred_list = fallback

            predictions.append({"Id": row["Id"], "Answer": format_answer(pred_list)})
            raw_file.write(
                json.dumps(
                    {
                        "Id": row["Id"],
                        "captions": captions,
                        "model_output": output_text,
                        "parsed_submission_answer": pred_list,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_submission(predictions, args.output, sample_df=sample_df)
    print(f"Saved submission to {args.output}")
    print(f"Saved raw outputs to {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
