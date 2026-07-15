from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.caption_augmented.config import DEFAULT_ORDER_MODEL
from src.caption_augmented.dataset import (
    OrderTrainingDataset,
    OrderTrainingRecord,
    build_records_from_dataframe,
    build_validation_records,
    load_filtered_train_dataframe,
    split_train_val_dataframe,
)
from src.caption_augmented.model import load_qwen_processor_and_model, _generate_qwen_text
from src.caption_augmented.prompts import build_order_messages
from src.submission import normalize_permutation, parse_permutation_from_text

DEFAULT_OUTPUT_DIR = "outputs/caption_augmented/orderer_train_smoke"
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the caption-augmented Qwen3.5/Qwen3-VL orderer.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-csv", default=None, help="Optional filtered train CSV; images are still read from data-dir/train")
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--missing-caption-policy", choices=["empty", "fail"], default="empty")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Model device_map. Use `local` for torchrun QLoRA, `none` for full/bf16 DDP, or a transformers device_map string.",
    )
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--use-lora", dest="use_lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--drop-no-ordering", action="store_true")
    parser.add_argument(
        "--shuffle-augmentations-per-sample",
        type=int,
        default=0,
        help="Create N shuffled image-order views per training row before building records. 0 disables this.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-include-identity", action="store_true")
    parser.add_argument("--shuffle-no-ordering", action="store_true")
    parser.add_argument("--shuffle-keep-original", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build records and write config without loading a model")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of rows (0 < f < 1) held out as validation, split before any shuffle "
        "augmentation. 0 (default) disables validation entirely, matching prior behavior.",
    )
    parser.add_argument("--val-seed", type=int, default=42, help="Seed for the deterministic train/val split.")
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=None,
        help="Cap the number of validation rows actually evaluated each epoch (cheap dry runs).",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens when generating validation predictions.",
    )
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


def write_training_preview(records: list[OrderTrainingRecord], output_dir: Path, *, dry_run: bool) -> None:
    if not is_main_process():
        return
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
    (output_dir / "orderer_training_preview.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_training_config(args: argparse.Namespace, records: list[OrderTrainingRecord]) -> None:
    if not is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["records"] = len(records)
    (output_dir / "orderer_training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_model_kwargs(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(torch, args.torch_dtype),
    }
    device_map = resolve_device_map(args.device_map)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA requires `bitsandbytes` on the training server.") from exc
        compute_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return model_kwargs


def resolve_device_map(value: str) -> str | dict[str, int] | None:
    if value == "none":
        return None
    if value == "local":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return {"": local_rank}
    return value


def load_orderer_training_bundle(args: argparse.Namespace):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Orderer training requires torch and a recent transformers build.") from exc

    processor, model = load_qwen_processor_and_model(args.model_name, build_model_kwargs(args, torch))
    return torch, processor, model


def parse_lora_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_lora(model: Any, args: argparse.Namespace) -> Any:
    if args.load_in_4bit and not args.use_lora:
        raise ValueError("4-bit training should be used with LoRA. Remove --no-lora or pass --no-load-in-4bit.")
    if not args.use_lora:
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("LoRA training requires `peft` on the training server.") from exc

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


class OrdererSFTCollator:
    def __init__(self, processor: Any, image_dir: Path):
        self.processor = processor
        self.image_dir = image_dir

    def _encode(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> dict[str, Any]:
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[OrderTrainingRecord]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("OrdererSFTCollator supports batch size 1; use gradient accumulation.")

        record = features[0]
        row = pd.Series(record.row)
        prompt_messages = build_order_messages(row, self.image_dir, record.captions)
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


def build_training_arguments(args: argparse.Namespace, torch: Any):
    from transformers import TrainingArguments

    cuda_available = torch.cuda.is_available()
    use_fp16 = cuda_available and args.torch_dtype == "float16"
    use_bf16 = cuda_available and args.torch_dtype == "bfloat16"
    save_strategy = "steps" if args.save_steps > 0 else "no"
    ddp_find_unused_parameters = False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None
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
        ddp_find_unused_parameters=ddp_find_unused_parameters,
    )


def parse_ground_truth_order(target_text: str) -> list[int]:
    return normalize_permutation(ast.literal_eval(target_text))


def evaluate_validation_records(
    model: Any,
    processor: Any,
    val_records: list[OrderTrainingRecord],
    image_dir: Path,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate a whole-order prediction for each validation record and score
    exact-match accuracy against its ground-truth chronological order. Mirrors
    the whole-mode inference path (`build_order_messages` + greedy generation),
    so these numbers are a fair local preview of `caption_augmented.infer`
    (whole mode) performance without needing a Kaggle submission.
    """
    was_training = model.training
    model.eval()
    correct = 0
    per_sample: list[dict[str, Any]] = []
    try:
        for record in val_records:
            row = pd.Series(record.row)
            messages = build_order_messages(row, image_dir=image_dir, captions=record.captions)
            output_text = _generate_qwen_text(processor, model, messages, max_new_tokens=max_new_tokens)
            ground_truth = parse_ground_truth_order(record.target_text)
            try:
                predicted = normalize_permutation(parse_permutation_from_text(output_text))
                is_correct = predicted == ground_truth
            except ValueError:
                predicted = None
                is_correct = False
            if is_correct:
                correct += 1
            per_sample.append(
                {
                    "Id": record.row.get("Id"),
                    "ground_truth": ground_truth,
                    "model_output": output_text,
                    "predicted": predicted,
                    "correct": is_correct,
                }
            )
    finally:
        if was_training:
            model.train()

    total = len(val_records)
    return {
        "exact_match_accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
    }


def build_validation_callback(
    processor: Any,
    val_records: list[OrderTrainingRecord],
    image_dir: Path,
    output_dir: Path,
    max_new_tokens: int,
):
    """Build a TrainerCallback that runs generation-based exact-match
    validation at the end of every epoch and checkpoints the best-scoring
    model/adapter to `output_dir/best`, separately from Trainer's own
    step-based checkpoints. `transformers` is imported lazily here (not at
    module level) so this module stays importable without torch/transformers
    installed, matching the rest of this file's lazy-import convention.
    """
    from transformers import TrainerCallback

    class ValidationCallback(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[dict[str, Any]] = []
            self.best_exact_match: float = -1.0

        def on_epoch_end(self, args, state, control, **kwargs):
            if not is_main_process():
                return control

            model = kwargs["model"]
            result = evaluate_validation_records(
                model, processor, val_records, image_dir, max_new_tokens=max_new_tokens
            )
            entry = {
                "epoch": state.epoch,
                "step": state.global_step,
                "exact_match_accuracy": result["exact_match_accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
            self.history.append(entry)
            print(
                f"[validation] epoch={entry['epoch']:.2f} step={entry['step']} "
                f"exact_match={result['exact_match_accuracy']:.4f} "
                f"({result['correct']}/{result['total']})"
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "validation_history.json").write_text(
                json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if result["exact_match_accuracy"] > self.best_exact_match:
                self.best_exact_match = result["exact_match_accuracy"]
                best_dir = output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(best_dir))
                processor.save_pretrained(str(best_dir))
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                (best_dir / "val_predictions.json").write_text(
                    json.dumps(result["per_sample"], indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"[validation] new best exact_match={self.best_exact_match:.4f} -> saved to {best_dir}")
            return control

    return ValidationCallback()


def main() -> int:
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError("--per-device-train-batch-size must be 1; increase --gradient-accumulation-steps instead.")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_df = load_filtered_train_dataframe(
        data_dir,
        args.train_csv,
        drop_no_ordering=args.drop_no_ordering,
        max_samples=args.max_samples,
    )

    val_records: list[OrderTrainingRecord] = []
    if args.val_fraction > 0:
        train_df, val_df = split_train_val_dataframe(train_df, args.val_fraction, seed=args.val_seed)
        if args.val_max_samples is not None:
            val_df = val_df.head(args.val_max_samples).copy()
        val_records = build_validation_records(
            val_df, args.caption_cache, missing_caption_policy=args.missing_caption_policy
        )

    records = build_records_from_dataframe(
        train_df,
        args.caption_cache,
        missing_caption_policy=args.missing_caption_policy,
        shuffle_augmentations_per_sample=args.shuffle_augmentations_per_sample,
        shuffle_seed=args.shuffle_seed,
        shuffle_include_identity=args.shuffle_include_identity,
        shuffle_no_ordering=args.shuffle_no_ordering,
        shuffle_keep_original=args.shuffle_keep_original,
    )
    write_training_preview(records, output_dir, dry_run=args.dry_run)
    write_training_config(args, records)
    if val_records and is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation_set.json").write_text(
            json.dumps([{"Id": record.row.get("Id")} for record in val_records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.dry_run:
        return 0

    torch, processor, model = load_orderer_training_bundle(args)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = configure_lora(model, args)

    from transformers import Trainer

    callbacks = []
    if val_records:
        callbacks.append(
            build_validation_callback(
                processor=processor,
                val_records=val_records,
                image_dir=data_dir / "train",
                output_dir=output_dir,
                max_new_tokens=args.eval_max_new_tokens,
            )
        )

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args, torch),
        train_dataset=OrderTrainingDataset(records),
        data_collator=OrdererSFTCollator(processor=processor, image_dir=data_dir / "train"),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    summary = {
        "status": "ok",
        "records": len(records),
        "val_records": len(val_records),
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "use_lora": args.use_lora,
        "load_in_4bit": args.load_in_4bit,
    }
    if is_main_process():
        (output_dir / "orderer_training_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())