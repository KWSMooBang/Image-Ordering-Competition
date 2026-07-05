from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.data_utils import INPUT_COLUMNS, read_csv
from src.submission import format_answer, normalize_permutation, parse_model_output, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen2-VL zero-shot baseline from baseline_code.ipynb.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/qwen2vl_submission.csv")
    parser.add_argument("--raw-output", default="outputs/qwen2vl_raw_outputs.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for smoke tests")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fallback-answer", default="[1, 2, 3, 4]")
    return parser.parse_args()


def get_prompt_message(row: pd.Series, image_dir: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, column in enumerate(INPUT_COLUMNS, start=1):
        img_path = image_dir / str(row["Id"]) / str(row[column])
        content.append({"type": "image", "image": str(img_path)})
        content.append({"type": "text", "text": f"\nImage {index}\n"})

    sentence = row["Sentence"]
    user_text = (
        f'Thinking about the sentence: "{sentence}"\n'
        "Look at the 4 images above labeled Image 1 to Image 4. "
        "Determine the correct chronological order of these images to match the sentence. "
        "Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    )
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def load_qwen(model_name: str):
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen baseline dependencies are missing. Run `bash init.sh` or install requirements.txt."
        ) from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return torch, process_vision_info, processor, model


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

    print(f"Loading model: {args.model_name}")
    torch, process_vision_info, processor, model = load_qwen(args.model_name)
    print(f"Running inference on {len(test_df)} samples")

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, str]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            messages = get_prompt_message(row, image_dir)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            try:
                pred_list = parse_model_output(output_text, fallback=fallback)
            except ValueError:
                pred_list = fallback

            predictions.append({"Id": row["Id"], "Answer": format_answer(pred_list)})
            raw_file.write(
                json.dumps(
                    {
                        "Id": row["Id"],
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
