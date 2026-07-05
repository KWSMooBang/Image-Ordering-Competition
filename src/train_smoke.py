from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageStat

from src.data_utils import INPUT_COLUMNS, image_paths_for_row, read_csv
from src.submission import parse_answer_cell


LABEL_SPACE = list(itertools.permutations([1, 2, 3, 4]))
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_SPACE)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny train step to validate data, torch, device, and checkpoint writing."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/train_smoke")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def image_mean_features(path: Path, image_size: int) -> list[float]:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        stat = ImageStat.Stat(image)
    return [value / 255.0 for value in stat.mean]


def row_features(row, image_root: Path, image_size: int) -> list[float]:
    features: list[float] = []
    for path in image_paths_for_row(row, image_root):
        features.extend(image_mean_features(path, image_size))

    sentence = str(row["Sentence"])
    features.append(min(len(sentence) / 512.0, 4.0))
    features.append(min(len(sentence.split()) / 96.0, 4.0))
    return features


def build_tensors(data_dir: Path, max_samples: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    train_df = read_csv(data_dir / "train.csv").head(max_samples)
    image_root = data_dir / "train"

    features = []
    labels = []
    for _, row in train_df.iterrows():
        answer = tuple(parse_answer_cell(row["Answer"]))
        features.append(row_features(row, image_root, image_size))
        labels.append(LABEL_TO_INDEX[answer])

    if not features:
        raise ValueError("No training rows were loaded for the smoke test.")

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    features, labels = build_tensors(data_dir, args.max_samples, args.image_size)
    features = features.to(device)
    labels = labels.to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(LABEL_SPACE)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    model.train()
    for _ in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    checkpoint_path = output_dir / "smoke_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "label_space": LABEL_SPACE,
            "losses": losses,
            "device": str(device),
        },
        checkpoint_path,
    )

    summary = {
        "status": "ok",
        "samples": int(features.shape[0]),
        "input_dim": int(features.shape[1]),
        "steps": args.max_steps,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "final_loss": losses[-1],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
