"""Train the pairwise image-ordering model.

Example:
    python -m src.training.train_pairwise \
        --config configs/pairwise_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PairwiseDataset, build_hf_image_transform
from src.evaluation.metrics import (
    binary_log_loss,
    cycle_rate,
    exact_match_accuracy,
    kendall_distance,
    pairwise_accuracy,
)
from src.inference.reconstruct_order import reconstruct_best_order
from src.models import PairwiseOrderingModel


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def tokenize_sentences(tokenizer: Any, sentences: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def build_loaders(config: dict[str, Any], image_transform: Any) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    training_config = config["training"]

    train_dataset = PairwiseDataset(
        pairs=data_config["train_pairs"],
        image_root=data_config["train_image_root"],
        transform=image_transform,
        swap_probability=float(training_config.get("swap_probability", 0.5)),
    )
    val_dataset = PairwiseDataset(
        pairs=data_config["val_pairs"],
        image_root=data_config["val_image_root"],
        transform=image_transform,
        swap_probability=0.0,
    )

    common = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and device.type == "cuda",
    )

    for batch in tqdm(loader, desc="train", leave=False):
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        text_inputs = tokenize_sentences(tokenizer, list(batch["sentence"]), device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(
                image_a=image_a,
                image_b=image_b,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return total_loss / max(sample_count, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    sample_count = 0
    records: list[dict] = []

    for batch in tqdm(loader, desc="val", leave=False):
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        text_inputs = tokenize_sentences(tokenizer, list(batch["sentence"]), device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(
                image_a=image_a,
                image_b=image_b,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )
            loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits).cpu().tolist()
        true_labels = labels.cpu().tolist()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

        for index in range(batch_size):
            records.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "pair_id": str(batch["pair_id"][index]),
                    "image_a_index": int(batch["image_a_index"][index]),
                    "image_b_index": int(batch["image_b_index"][index]),
                    "label": int(true_labels[index]),
                    "probability": float(probabilities[index]),
                }
            )

    predictions = pd.DataFrame.from_records(records)
    y_true = predictions["label"].tolist()
    y_probability = predictions["probability"].tolist()

    true_orders: list[list[int]] = []
    predicted_orders: list[list[int]] = []
    probability_groups = []

    for _, group in predictions.groupby("sample_id", sort=False):
        true_probabilities = {
            (int(row.image_a_index), int(row.image_b_index)): float(row.label)
            for row in group.itertuples(index=False)
        }
        predicted_probabilities = {
            (int(row.image_a_index), int(row.image_b_index)): float(row.probability)
            for row in group.itertuples(index=False)
        }
        true_order, _ = reconstruct_best_order(true_probabilities)
        predicted_order, _ = reconstruct_best_order(predicted_probabilities)

        true_orders.append(true_order)
        predicted_orders.append(predicted_order)
        probability_groups.append(predicted_probabilities)

    mean_kendall = (
        sum(
            kendall_distance(true_order, predicted_order)
            for true_order, predicted_order in zip(true_orders, predicted_orders)
        )
        / len(true_orders)
        if true_orders
        else 0.0
    )

    metrics = {
        "val_loss": total_loss / max(sample_count, 1),
        "pairwise_accuracy": pairwise_accuracy(
            y_true, y_probability, threshold=threshold
        ),
        "pairwise_log_loss": binary_log_loss(y_true, y_probability),
        "exact_match_accuracy": exact_match_accuracy(
            true_orders, predicted_orders
        ),
        "mean_kendall_distance": mean_kendall,
        "cycle_rate": cycle_rate(
            probability_groups,
            threshold=threshold,
        ),
    }
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise ordering model.")
    parser.add_argument(
        "--config",
        default="configs/pairwise_baseline.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("device", "auto")))
    model_config = config["model"]
    training_config = config["training"]
    inference_config = config.get("inference", {})

    experiment_name = config["experiment"]["name"]
    experiment_dir = (
        Path(config["experiment"].get("output_dir", "outputs/experiments"))
        / experiment_name
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, experiment_dir / "config.yaml")

    backbone_name = model_config["backbone"]
    image_transform = build_hf_image_transform(backbone_name)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install it with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    train_loader, val_loader = build_loaders(config, image_transform)

    model = PairwiseOrderingModel(
        backbone_name=backbone_name,
        projection_dim=int(model_config.get("projection_dim", 256)),
        hidden_dim=int(model_config.get("hidden_dim", 512)),
        dropout=float(model_config.get("dropout", 0.2)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )

    epochs = int(training_config["epochs"])
    patience = int(training_config.get("early_stopping_patience", 3))
    use_amp = bool(training_config.get("use_amp", True))
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    threshold = float(inference_config.get("threshold", 0.5))

    best_exact_match = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print(f"[train] device={device}")
    print(f"[train] experiment={experiment_name}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            tokenizer,
            optimizer,
            criterion,
            device,
            use_amp=use_amp,
            gradient_clip_norm=gradient_clip_norm,
        )
        metrics, predictions = validate(
            model,
            val_loader,
            tokenizer,
            criterion,
            device,
            use_amp=use_amp,
            threshold=threshold,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"pair_acc={metrics['pairwise_accuracy']:.4f} "
            f"exact_acc={metrics['exact_match_accuracy']:.4f} "
            f"kendall={metrics['mean_kendall_distance']:.4f}"
        )

        exact_match = metrics["exact_match_accuracy"]
        if exact_match > best_exact_match:
            best_exact_match = exact_match
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "metrics": metrics,
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            predictions.to_csv(
                experiment_dir / "val_predictions.csv",
                index=False,
            )
            with (experiment_dir / "metrics.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metrics, file, ensure_ascii=False, indent=2)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[train] early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(experiment_dir / "history.csv", index=False)
    print(f"[train] best_exact_match={best_exact_match:.4f}")
    print(f"[train] output={experiment_dir.resolve()}")


if __name__ == "__main__":
    main()
