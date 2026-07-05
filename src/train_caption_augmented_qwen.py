from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.caption_augmented_common import get_order_message, load_caption_cache
from src.data_utils import INPUT_COLUMNS, read_csv
from src.qwen_vl_common import DEFAULT_MODEL_NAME, detect_qwen_family, get_processor_kwargs
from src.submission import parse_answer_cell, submission_to_chronological

DEFAULT_OUTPUT_DIR = "checkpoints/caption_augmented_qwen_lora"
DEFAULT_TRAIN_MAX_PIXELS = 768 * 28 * 28
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class TrainingRecord:
    row: dict[str, Any]
    captions: list[str]
    target_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA SFT for the caption-augmented Qwen VL ordering prompt."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--caption-cache", default=None, help="Optional JSONL caption cache for train images")
    parser.add_argument("--missing-caption-policy", choices=["empty", "fail"], default="empty")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional attention backend passed to from_pretrained.",
    )
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_TRAIN_MAX_PIXELS,
        help="Per-image processor pixel budget. Lower this first if a 24GB GPU OOMs.",
    )
    parser.add_argument("--use-lora", dest="use_lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--drop-no-ordering", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build records and write a summary without importing transformers or loading a model.",
    )
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


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def target_text_from_answer(answer: object) -> str:
    submission_answer = parse_answer_cell(answer)
    chronological_order = submission_to_chronological(submission_answer)
    return str(chronological_order)


def captions_for_row(
    row: pd.Series,
    caption_cache: dict[tuple[str, int, str], str],
    missing_caption_policy: str,
) -> list[str]:
    captions: list[str] = []
    missing: list[str] = []
    for image_index, column in enumerate(INPUT_COLUMNS, start=1):
        image_name = str(row[column])
        key = (str(row["Id"]), image_index, image_name)
        caption = caption_cache.get(key)
        if caption is None:
            missing.append(f"Id={row['Id']} image_index={image_index} image={image_name}")
            caption = ""
        captions.append(caption)

    if missing and missing_caption_policy == "fail":
        preview = "; ".join(missing[:5])
        raise ValueError(f"Missing cached captions, e.g. {preview}")
    return captions


def build_training_records(
    data_dir: Path,
    caption_cache_path: str | Path | None,
    missing_caption_policy: str,
    max_samples: int | None,
    drop_no_ordering: bool,
) -> list[TrainingRecord]:
    train_df = read_csv(data_dir / "train.csv")
    if drop_no_ordering:
        train_df = train_df[~train_df["No_ordering"].map(is_truthy)]
    if max_samples is not None:
        train_df = train_df.head(max_samples).copy()

    caption_cache = load_caption_cache(caption_cache_path) if caption_cache_path else {}
    records: list[TrainingRecord] = []
    for _, row in train_df.iterrows():
        records.append(
            TrainingRecord(
                row=row.to_dict(),
                captions=captions_for_row(row, caption_cache, missing_caption_policy),
                target_text=target_text_from_answer(row["Answer"]),
            )
        )

    if not records:
        raise ValueError("No training records were built.")
    return records


def summarize_records(records: list[TrainingRecord], output_dir: Path, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first = records[0]
    summary = {
        "status": "dry_run_ok" if dry_run else "records_built",
        "records": len(records),
        "first_record": {
            "Id": first.row["Id"],
            "target_text": first.target_text,
            "captions": first.captions,
        },
    }
    (output_dir / "train_records_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


class CaptionAugmentedOrderingDataset:
    def __init__(self, records: list[TrainingRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrainingRecord:
        return self.records[index]


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_training_bundle(args: argparse.Namespace) -> tuple[Any, Any, Any, str, Any | None]:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Training requires torch and transformers.") from exc

    family = detect_qwen_family(args.model_name)
    model_dtype = resolve_torch_dtype(torch, args.torch_dtype)
    processor_kwargs = get_processor_kwargs(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires `bitsandbytes`. Install GPU training dependencies on the 3090 server."
            ) from exc
        compute_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    process_vision_info = None
    if family == "qwen3-vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL training requires a recent transformers build.") from exc
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    elif family == "qwen2.5-vl":
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen2.5-VL training requires qwen-vl-utils and recent transformers.") from exc
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    else:
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen2-VL training requires qwen-vl-utils and transformers.") from exc
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)

    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    return torch, processor, model, family, process_vision_info


def parse_lora_target_modules(value: str) -> list[str]:
    return [module.strip() for module in value.split(",") if module.strip()]


def configure_lora(model: Any, args: argparse.Namespace) -> Any:
    if args.load_in_4bit and not args.use_lora:
        raise ValueError("4-bit training should be used with LoRA. Remove --no-lora or pass --no-load-in-4bit.")
    if not args.use_lora:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("LoRA training requires `peft`. Install GPU training dependencies.") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=parse_lora_target_modules(args.lora_target_modules),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


class SFTDataCollator:
    def __init__(
        self,
        processor: Any,
        family: str,
        image_dir: Path,
        process_vision_info: Any | None,
    ):
        self.processor = processor
        self.family = family
        self.image_dir = image_dir
        self.process_vision_info = process_vision_info

    def _encode(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> dict[str, Any]:
        if self.family == "qwen3-vl":
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
            )

        if self.process_vision_info is None:
            raise RuntimeError("Qwen2-style collation requires process_vision_info.")
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[TrainingRecord]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("This multimodal collator currently supports batch size 1; use gradient accumulation.")

        record = features[0]
        row = pd.Series(record.row)
        prompt_messages = get_order_message(row, self.image_dir, record.captions)
        full_messages = prompt_messages + [{"role": "assistant", "content": record.target_text}]

        prompt_inputs = self._encode(prompt_messages, add_generation_prompt=True)
        full_inputs = self._encode(full_messages, add_generation_prompt=False)
        labels = full_inputs["input_ids"].clone()

        prompt_length = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_length] = -100
        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is not None:
            labels[full_inputs["input_ids"] == pad_token_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


def build_training_arguments(args: argparse.Namespace, torch: Any) -> Any:
    from transformers import TrainingArguments

    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and args.torch_dtype == "bfloat16"
    use_fp16 = cuda_available and args.torch_dtype == "float16"
    save_strategy = "steps" if args.save_steps > 0 else "no"

    return TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=max(args.save_steps, 1),
        save_strategy=save_strategy,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
    )


def write_training_config(args: argparse.Namespace, records: list[TrainingRecord]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["records"] = len(records)
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError("--per-device-train-batch-size must be 1; increase --gradient-accumulation-steps instead.")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    records = build_training_records(
        data_dir=data_dir,
        caption_cache_path=args.caption_cache,
        missing_caption_policy=args.missing_caption_policy,
        max_samples=args.max_samples,
        drop_no_ordering=args.drop_no_ordering,
    )
    summarize_records(records, output_dir=output_dir, dry_run=args.dry_run)
    write_training_config(args, records)
    if args.dry_run:
        return 0

    torch, processor, model, family, process_vision_info = load_training_bundle(args)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = configure_lora(model, args)

    dataset = CaptionAugmentedOrderingDataset(records)
    collator = SFTDataCollator(
        processor=processor,
        family=family,
        image_dir=data_dir / "train",
        process_vision_info=process_vision_info,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args, torch),
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    summary = {
        "status": "ok",
        "output_dir": args.output_dir,
        "records": len(records),
        "model_name": args.model_name,
        "use_lora": args.use_lora,
        "load_in_4bit": args.load_in_4bit,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
