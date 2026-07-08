from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data_filtering.quality import (
    DataFilteringConfig,
    build_audit_frame,
    filter_train_frame,
    load_caption_cache,
)
from src.data_filtering.siglip import DEFAULT_SIGLIP_MODEL, SiglipImageTextScorer
from src.data_utils import read_csv

DEFAULT_OUTPUT = "outputs/data_filtering/train_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit train samples and build a data-filtering manifest.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--filtered-output", default=None)
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument(
        "--relevance-backend",
        choices=["auto", "none", "siglip"],
        default="auto",
        help="Frame-text relevance backend. Use siglip to compare images and optional cached captions with the sentence.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--drop-actions", default="drop_from_supervised")
    parser.add_argument("--keep-no-ordering", dest="drop_no_ordering", action="store_false")
    parser.set_defaults(drop_no_ordering=True)

    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--siglip-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--siglip-batch-size", type=int, default=4)

    parser.add_argument("--dark-mean-max", type=float, default=DataFilteringConfig.dark_mean_max)
    parser.add_argument("--bright-mean-min", type=float, default=DataFilteringConfig.bright_mean_min)
    parser.add_argument("--blank-std-max", type=float, default=DataFilteringConfig.blank_std_max)
    parser.add_argument("--flat-std-max", type=float, default=DataFilteringConfig.flat_std_max)
    parser.add_argument("--flat-entropy-max", type=float, default=DataFilteringConfig.flat_entropy_max)
    parser.add_argument("--duplicate-hash-distance", type=int, default=DataFilteringConfig.duplicate_hash_distance)
    parser.add_argument("--duplicate-mean-delta", type=float, default=DataFilteringConfig.duplicate_mean_delta)
    parser.add_argument(
        "--downweight-blank-frame-count",
        type=int,
        default=DataFilteringConfig.downweight_blank_frame_count,
    )
    parser.add_argument("--drop-blank-frame-count", type=int, default=DataFilteringConfig.drop_blank_frame_count)
    parser.add_argument("--low-relevance-threshold", type=float, default=DataFilteringConfig.low_relevance_threshold)
    parser.add_argument("--low-relevance-frame-count", type=int, default=DataFilteringConfig.low_relevance_frame_count)
    parser.add_argument("--clean-weight", type=float, default=DataFilteringConfig.clean_weight)
    parser.add_argument("--downweight-weight", type=float, default=DataFilteringConfig.downweight_weight)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DataFilteringConfig:
    return DataFilteringConfig(
        dark_mean_max=args.dark_mean_max,
        bright_mean_min=args.bright_mean_min,
        blank_std_max=args.blank_std_max,
        flat_std_max=args.flat_std_max,
        flat_entropy_max=args.flat_entropy_max,
        duplicate_hash_distance=args.duplicate_hash_distance,
        duplicate_mean_delta=args.duplicate_mean_delta,
        drop_no_ordering=args.drop_no_ordering,
        downweight_blank_frame_count=args.downweight_blank_frame_count,
        drop_blank_frame_count=args.drop_blank_frame_count,
        low_relevance_threshold=args.low_relevance_threshold,
        low_relevance_frame_count=args.low_relevance_frame_count,
        clean_weight=args.clean_weight,
        downweight_weight=args.downweight_weight,
    )


def parse_drop_actions(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_relevance_backend(args: argparse.Namespace) -> str:
    if args.relevance_backend != "auto":
        return args.relevance_backend
    return "none"


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_df = read_csv(data_dir / "train.csv")
    if args.max_samples is not None:
        train_df = train_df.head(args.max_samples).copy()

    relevance_backend = resolve_relevance_backend(args)
    caption_cache = None
    relevance_scorer = None
    if relevance_backend == "siglip":
        caption_cache = load_caption_cache(args.caption_cache) if args.caption_cache else None
        relevance_scorer = SiglipImageTextScorer(
            model_name=args.siglip_model,
            device=args.siglip_device,
            torch_dtype=args.siglip_torch_dtype,
            batch_size=args.siglip_batch_size,
        )

    audit_df = build_audit_frame(
        train_df,
        data_dir / "train",
        config=config_from_args(args),
        caption_cache=caption_cache,
        relevance_scorer=relevance_scorer,
        relevance_backend=relevance_backend,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)

    summary = {
        "rows": len(audit_df),
        "output": str(output_path),
        "actions": audit_df["action"].value_counts().sort_index().to_dict(),
        "manual_review": int(audit_df["manual_review"].sum()),
        "blank_frame_rows": int((audit_df["blank_frame_count"] > 0).sum()),
        "duplicate_candidate_rows": int((audit_df["duplicate_pair_count"] > 0).sum()),
        "low_relevance_rows": int((audit_df["low_relevance_frame_count"] > 0).sum()),
    }

    if args.filtered_output:
        filtered = filter_train_frame(train_df, audit_df, drop_actions=parse_drop_actions(args.drop_actions))
        filtered_output = Path(args.filtered_output)
        filtered_output.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(filtered_output, index=False)
        summary["filtered_output"] = str(filtered_output)
        summary["filtered_rows"] = len(filtered)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
