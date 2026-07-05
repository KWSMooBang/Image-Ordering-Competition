from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from src.caption_augmented_common import get_or_generate_captions, get_order_message, load_caption_cache
from src.data_utils import read_csv
from src.qwen_vl_common import DEFAULT_MODEL_NAME, generate_text, load_qwen_vl
from src.submission import format_answer, normalize_permutation, parse_model_output, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-image captions with a Qwen VLM, then run ordering with "
            "both images and captions."
        )
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/caption_augmented_qwen_submission.csv")
    parser.add_argument("--raw-output", default="outputs/caption_augmented_qwen_raw_outputs.jsonl")
    parser.add_argument("--caption-cache", default="outputs/caption_augmented_qwen_captions.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Ordering VLM")
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT/LoRA adapter for the ordering VLM")
    parser.add_argument(
        "--caption-model-name",
        default=None,
        help="Captioning VLM. Defaults to --model-name and reuses the same loaded model.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for smoke tests")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--caption-max-new-tokens", type=int, default=80)
    parser.add_argument("--max-caption-chars", type=int, default=360)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional attention backend passed to from_pretrained. Try flash_attention_2 on CUDA if installed.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional processor min_pixels budget for each image.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Optional processor max_pixels budget for each image. Lower this first if a 24GB GPU OOMs.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base VLM with bitsandbytes 4-bit quantization")
    parser.add_argument("--fallback-answer", default="[1, 2, 3, 4]")
    parser.add_argument("--refresh-captions", action="store_true", help="Ignore cached captions and regenerate them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    test_df = read_csv(data_dir / "test.csv")
    sample_df = read_csv(data_dir / "sample_submission.csv")
    image_dir = data_dir / "test"
    fallback = normalize_permutation(ast.literal_eval(args.fallback_answer))

    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples).copy()
        sample_df = sample_df.head(args.max_samples).copy()

    caption_model_name = args.caption_model_name or args.model_name
    print(f"Loading ordering model: {args.model_name}")
    ordering_bundle = load_qwen_vl(
        args.model_name,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
        adapter_path=args.adapter_path,
    )
    if caption_model_name == args.model_name:
        caption_bundle = ordering_bundle
        print("Reusing ordering model for caption generation")
    else:
        print(f"Loading caption model: {caption_model_name}")
        caption_bundle = load_qwen_vl(
            caption_model_name,
            attn_implementation=args.attn_implementation,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            load_in_4bit=args.load_in_4bit,
        )

    caption_cache_path = Path(args.caption_cache)
    caption_cache_path.parent.mkdir(parents=True, exist_ok=True)
    caption_cache = load_caption_cache(caption_cache_path)

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running caption-augmented inference on {len(test_df)} samples")
    predictions: list[dict[str, str]] = []
    with caption_cache_path.open("a", encoding="utf-8") as caption_handle, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_file:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            captions = get_or_generate_captions(
                row=row,
                image_dir=image_dir,
                bundle=caption_bundle,
                cache=caption_cache,
                cache_handle=caption_handle,
                caption_max_new_tokens=args.caption_max_new_tokens,
                max_caption_chars=args.max_caption_chars,
                refresh_captions=args.refresh_captions,
            )
            messages = get_order_message(row, image_dir, captions)
            output_text = generate_text(ordering_bundle, messages, max_new_tokens=args.max_new_tokens)

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
    print(f"Saved caption cache to {caption_cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
