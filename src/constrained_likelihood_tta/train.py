from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.constrained_likelihood_tta.config import DEFAULT_ORDER_MODEL
from src.constrained_likelihood_tta.dataset import (
    OrderTrainingDataset,
    OrderTrainingRecord,
    build_training_records,
)
from src.constrained_likelihood_tta.model import load_qwen_bundle
from src.constrained_likelihood_tta.prompts import build_order_messages

DEFAULT_OUTPUT_DIR = "checkpoints/constrained_likelihood_tta/qwen35_4b_lora"
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the caption-augmented orderer used by constrained likelihood TTA."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument(
        "--caption-missing-policy", choices=["empty", "fail"], default="fail"
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument(
        "--load-in-4bit", dest="load_in_4bit", action="store_true", default=True
    )
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument(
        "--use-lora", dest="use_lora", action="store_true", default=True
    )
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--shuffle-augmentations-per-sample", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-keep-original", action="store_true", default=True)
    parser.add_argument(
        "--shuffle-only", dest="shuffle_keep_original", action="store_false"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def resolve_device_map(value: str) -> str | dict[str, int] | None:
    if value == "none":
        return None
    if value == "local":
        return {"": int(os.environ.get("LOCAL_RANK", "0"))}
    return value


def parse_lora_targets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_quantization_config(args: argparse.Namespace, torch: Any) -> Any:
    if not args.load_in_4bit:
        return None
    if not args.use_lora:
        raise ValueError("4-bit training requires LoRA")
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("4-bit QLoRA requires bitsandbytes") from exc
    compute_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def configure_lora(model: Any, args: argparse.Namespace) -> Any:
    if not args.use_lora:
        if args.gradient_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()
        return model
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError("LoRA training requires peft") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=parse_lora_targets(args.lora_target_modules),
        ),
    )
    model.print_trainable_parameters()
    return model


class OrdererSFTCollator:
    def __init__(self, processor: Any, image_dir: Path) -> None:
        self.processor = processor
        self.image_dir = image_dir

    def _encode(self, messages: list[dict[str, Any]], *, generation_prompt: bool):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[OrderTrainingRecord]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("OrdererSFTCollator requires batch size 1")
        record = features[0]
        prompt_messages = build_order_messages(
            pd.Series(record.row),
            self.image_dir,
            record.captions,
        )
        full_messages = prompt_messages + [
            {"role": "assistant", "content": record.target_text}
        ]
        prompt_inputs = self._encode(prompt_messages, generation_prompt=True)
        full_inputs = self._encode(full_messages, generation_prompt=False)
        labels = full_inputs["input_ids"].clone()
        prompt_length = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_length] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[full_inputs["input_ids"] == pad_token_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


def write_preflight(
    args: argparse.Namespace,
    records: list[OrderTrainingRecord],
    *,
    status: str,
) -> None:
    if not is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["records"] = len(records)
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    preview = {
        "status": status,
        "records": len(records),
        "first_record": {
            "Id": records[0].row["Id"],
            "target_text": records[0].target_text,
            "captions": records[0].captions,
        },
    }
    (output_dir / "training_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(preview, indent=2, ensure_ascii=False))


def build_training_arguments(args: argparse.Namespace, torch: Any):
    from transformers import TrainingArguments

    cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=max(args.save_steps, 1),
        save_strategy="steps" if args.save_steps > 0 else "no",
        fp16=cuda and args.torch_dtype == "float16",
        bf16=cuda and args.torch_dtype == "bfloat16",
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        ddp_find_unused_parameters=(
            False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None
        ),
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError(
            "per-device train batch size must be 1; use gradient accumulation"
        )
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    records = build_training_records(
        data_dir=data_dir,
        train_csv_path=args.train_csv,
        caption_cache_path=args.caption_cache,
        caption_missing_policy=args.caption_missing_policy,
        max_samples=args.max_samples,
        shuffle_augmentations_per_sample=args.shuffle_augmentations_per_sample,
        shuffle_seed=args.shuffle_seed,
        shuffle_keep_original=args.shuffle_keep_original,
    )
    write_preflight(
        args, records, status="dry_run_ok" if args.dry_run else "records_built"
    )
    if args.dry_run:
        return 0

    import torch

    processor, model = load_qwen_bundle(
        args.model_name,
        device_map=resolve_device_map(args.device_map),
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        quantization_config=build_quantization_config(args, torch),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = configure_lora(model, args)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args, torch),
        train_dataset=OrderTrainingDataset(records),
        data_collator=OrdererSFTCollator(processor, data_dir / "train"),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if is_main_process():
        summary = {
            "status": "ok",
            "records": len(records),
            "model_name": args.model_name,
            "output_dir": args.output_dir,
        }
        (Path(args.output_dir) / "training_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
