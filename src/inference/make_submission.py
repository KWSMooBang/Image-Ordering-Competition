"""Run pairwise inference on test data and create submission.csv.

Example:
    python -m src.inference.make_submission \
        --config configs/pairwise_baseline.yaml \
        --checkpoint outputs/experiments/exp001_pairwise_siglip/checkpoints/best.pt \
        --output outputs/submissions/submission_exp001.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PairwiseDataset, build_hf_image_transform
from src.inference.reconstruct_order import reconstruct_dataframe
from src.models import PairwiseOrderingModel


IMAGE_COLUMNS = ("Input_1", "Input_2", "Input_3", "Input_4")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_test_pairs(test_dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {"Id", "Sentence", *IMAGE_COLUMNS}
    missing = sorted(required - set(test_dataframe.columns))
    if missing:
        raise ValueError(f"Test CSV is missing columns: {missing}")

    records: list[dict] = []
    for row in test_dataframe.itertuples(index=False):
        sample_id = str(row.Id)
        files = {
            index: str(getattr(row, column))
            for index, column in enumerate(IMAGE_COLUMNS, start=1)
        }

        for image_a_index, image_b_index in combinations(range(1, 5), 2):
            records.append(
                {
                    "sample_id": sample_id,
                    "pair_id": f"{sample_id}__{image_a_index}_{image_b_index}",
                    "sentence": str(row.Sentence),
                    "image_a_index": image_a_index,
                    "image_b_index": image_b_index,
                    "image_a_path": (
                        PurePosixPath(sample_id) / files[image_a_index]
                    ).as_posix(),
                    "image_b_path": (
                        PurePosixPath(sample_id) / files[image_b_index]
                    ).as_posix(),
                    "label": 0,
                    "no_ordering": False,
                }
            )
    return pd.DataFrame.from_records(records)


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


@torch.no_grad()
def predict_pairs(
    model: PairwiseOrderingModel,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records: list[dict] = []

    for batch in tqdm(loader, desc="test inference"):
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        text_inputs = tokenize_sentences(tokenizer, list(batch["sentence"]), device)

        logits = model(
            image_a=image_a,
            image_b=image_b,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )
        probabilities = torch.sigmoid(logits).cpu().tolist()

        for index, probability in enumerate(probabilities):
            records.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "pair_id": str(batch["pair_id"][index]),
                    "image_a_index": int(batch["image_a_index"][index]),
                    "image_b_index": int(batch["image_b_index"][index]),
                    "probability": float(probability),
                }
            )

    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create competition submission.")
    parser.add_argument(
        "--config",
        default="configs/pairwise_baseline.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="outputs/submissions/submission_pairwise.csv",
    )
    parser.add_argument(
        "--save-pair-predictions",
        default=None,
        help="Optional CSV path for raw pair probabilities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(str(config.get("device", "auto")))

    data_config = config["data"]
    model_config = config["model"]
    inference_config = config["inference"]

    test_dataframe = pd.read_csv(data_config["test_csv"])
    test_pairs = build_test_pairs(test_dataframe)

    backbone_name = model_config["backbone"]
    image_transform = build_hf_image_transform(backbone_name)
    dataset = PairwiseDataset(
        pairs=test_pairs,
        image_root=data_config["test_image_root"],
        transform=image_transform,
        swap_probability=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(inference_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(inference_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers before inference.") from exc

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = PairwiseOrderingModel(
        backbone_name=backbone_name,
        projection_dim=int(model_config.get("projection_dim", 256)),
        hidden_dim=int(model_config.get("hidden_dim", 512)),
        dropout=float(model_config.get("dropout", 0.2)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pair_predictions = predict_pairs(model, loader, tokenizer, device)
    reconstructed = reconstruct_dataframe(pair_predictions)

    if args.save_pair_predictions:
        pair_path = Path(args.save_pair_predictions)
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_predictions.to_csv(pair_path, index=False)

    if data_config.get("sample_submission"):
        sample_path = Path(data_config["sample_submission"])
        if sample_path.exists():
            sample_submission = pd.read_csv(sample_path)
            submission = sample_submission[["Id"]].copy()
            submission["Id"] = submission["Id"].astype(str)
            reconstructed["Id"] = reconstructed["Id"].astype(str)
            submission = submission.merge(
                reconstructed[["Id", "Answer"]],
                on="Id",
                how="left",
                validate="one_to_one",
            )
        else:
            submission = reconstructed[["Id", "Answer"]].copy()
    else:
        submission = reconstructed[["Id", "Answer"]].copy()

    if submission["Answer"].isna().any():
        missing_ids = submission.loc[submission["Answer"].isna(), "Id"].head(10).tolist()
        raise RuntimeError(f"Missing reconstructed answers for IDs: {missing_ids}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"[submission] samples={len(submission):,}")
    print(f"[submission] saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
