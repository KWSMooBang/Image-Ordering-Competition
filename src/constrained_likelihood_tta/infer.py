from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from src.constrained_likelihood_tta.captions import captions_for_row, load_caption_cache
from src.constrained_likelihood_tta.config import DEFAULT_ORDER_MODEL, Defaults
from src.constrained_likelihood_tta.likelihood import (
    ALL_CANDIDATE_ORDERS,
    aggregate_candidate_scores,
    rank_candidate_scores,
    restore_candidate_scores,
)
from src.constrained_likelihood_tta.model import ConstrainedQwenOrderer
from src.constrained_likelihood_tta.prompts import build_order_messages
from src.constrained_likelihood_tta.tta import (
    build_tta_permutations,
    permute_row_and_captions,
)
from src.data_utils import read_csv
from src.submission import chronological_to_submission, format_answer, write_submission


def parse_args() -> argparse.Namespace:
    defaults = Defaults()
    parser = argparse.ArgumentParser(
        description="Run caption-augmented constrained likelihood decoding with permutation TTA."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=defaults.output)
    parser.add_argument("--raw-output", default=defaults.raw_output)
    parser.add_argument("--caption-cache", default=defaults.test_caption_cache)
    parser.add_argument(
        "--caption-missing-policy", choices=["empty", "fail"], default="fail"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--order-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--order-adapter", default=None)
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
        "--candidate-batch-size", type=int, default=defaults.candidate_batch_size
    )
    parser.add_argument("--score-normalization", choices=["sum", "mean"], default="sum")
    parser.add_argument(
        "--tta-permutations", type=int, default=defaults.tta_permutations
    )
    parser.add_argument("--tta-seed", type=int, default=42)
    parser.add_argument("--raw-top-k", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 2 <= args.candidate_batch_size <= 4:
        raise ValueError("candidate batch size must be between 2 and 4")
    if not 1 <= args.raw_top_k <= len(ALL_CANDIDATE_ORDERS):
        raise ValueError("raw top-k must be between 1 and 24")

    data_dir = Path(args.data_dir)
    test_df = read_csv(data_dir / "test.csv")
    sample_df = read_csv(data_dir / "sample_submission.csv")
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples).copy()
        sample_df = sample_df.head(args.max_samples).copy()

    caption_cache = load_caption_cache(args.caption_cache)
    tta_permutations = build_tta_permutations(args.tta_permutations, seed=args.tta_seed)
    orderer = ConstrainedQwenOrderer(
        args.order_model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        adapter_path=args.order_adapter,
    )

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    image_dir = data_dir / "test"

    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            captions = captions_for_row(
                row,
                caption_cache,
                missing_policy=args.caption_missing_policy,
            )
            restored_views: list[dict[tuple[int, int, int, int], float]] = []
            view_logs: list[dict[str, object]] = []
            for permutation in tta_permutations:
                permuted_row, permuted_captions = permute_row_and_captions(
                    row,
                    captions,
                    permutation,
                )
                messages = build_order_messages(
                    permuted_row, image_dir, permuted_captions
                )
                likelihoods = orderer.score_candidates(
                    messages,
                    ALL_CANDIDATE_ORDERS,
                    candidate_batch_size=args.candidate_batch_size,
                    normalization=args.score_normalization,
                )
                permuted_scores = {item.order: item.score for item in likelihoods}
                restored_scores = restore_candidate_scores(permuted_scores, permutation)
                restored_views.append(restored_scores)

                details = []
                for item in likelihoods:
                    restored_order = [
                        permutation[new_slot - 1] for new_slot in item.order
                    ]
                    details.append(
                        {
                            "permuted_chronological_order": list(item.order),
                            "restored_chronological_order": restored_order,
                            "log_likelihood": item.log_likelihood,
                            "token_count": item.token_count,
                            "score": item.score,
                        }
                    )
                details.sort(
                    key=lambda value: (
                        -float(value["score"]),
                        value["permuted_chronological_order"],
                    )
                )
                view_logs.append(
                    {
                        "permutation_new_slot_to_original": permutation,
                        "candidate_scores": details[: args.raw_top_k],
                    }
                )

            aggregated = aggregate_candidate_scores(restored_views)
            ranking = rank_candidate_scores(aggregated)
            best_order, best_score = ranking[0]
            margin = best_score - ranking[1][1]
            submission_answer = chronological_to_submission(best_order)
            predictions.append(
                {"Id": row["Id"], "Answer": format_answer(submission_answer)}
            )
            raw_handle.write(
                json.dumps(
                    {
                        "Id": row["Id"],
                        "captions": captions,
                        "decode_mode": "constrained_likelihood",
                        "score_normalization": args.score_normalization,
                        "candidate_batch_size": args.candidate_batch_size,
                        "tta_views": view_logs,
                        "aggregated_candidate_scores": [
                            {"chronological_order": list(order), "score": score}
                            for order, score in ranking[: args.raw_top_k]
                        ],
                        "winning_chronological_order": list(best_order),
                        "winning_score": best_score,
                        "score_margin": margin,
                        "submission_answer": submission_answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_submission(predictions, args.output, sample_df=sample_df)
    print(f"Saved submission to {args.output}")
    print(f"Saved candidate scores to {args.raw_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
