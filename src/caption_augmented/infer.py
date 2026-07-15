from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from src.caption_augmented.captions import (
    build_captioner,
    generate_fresh_captions_for_row,
)
from src.caption_augmented.config import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_ORDER_MODEL,
    CaptionAugmentedDefaults,
)
from src.caption_augmented.model import QwenOrderer
from src.caption_augmented.pairwise import (
    best_order_from_pairwise,
    build_pairwise_messages,
    collect_pairwise_judgments,
    find_disputed_pair,
    parse_pair_order_from_text,
    resolve_disputed_pair,
)
from src.caption_augmented.prompts import build_order_messages
from src.caption_augmented.tta import (
    build_tta_permutations,
    consensus_chronological_order,
    permute_row_and_captions,
    restore_chronological_order,
)
from src.data_utils import read_csv
from src.submission import (
    chronological_to_submission,
    format_answer,
    normalize_permutation,
    parse_permutation_from_text,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    defaults = CaptionAugmentedDefaults()
    parser = argparse.ArgumentParser(description="Run caption-augmented Qwen ordering inference.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=defaults.output)
    parser.add_argument("--raw-output", default=defaults.raw_output)
    parser.add_argument("--caption-cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fallback-answer", default="[1, 2, 3, 4]")

    parser.add_argument(
        "--caption-missing-policy",
        choices=["generate", "fail", "empty"],
        default="generate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--caption-backend", choices=["blip", "qwen"], default="blip")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--qwen-caption-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--caption-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--caption-torch-dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--caption-max-new-tokens", type=int, default=defaults.caption_max_new_tokens)
    parser.add_argument("--max-caption-chars", type=int, default=defaults.max_caption_chars)
    parser.add_argument("--sentence-aware-captions", action="store_true")
    parser.add_argument("--refresh-captions", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--order-model", default=DEFAULT_ORDER_MODEL)
    parser.add_argument("--order-adapter", default=None, help="Optional LoRA/PEFT adapter directory from training")
    parser.add_argument("--order-max-new-tokens", type=int, default=defaults.order_max_new_tokens)
    parser.add_argument(
        "--tta-permutations",
        type=int,
        default=4,
        help="Number of input permutations to ensemble (1 disables TTA, maximum 24)",
    )
    parser.add_argument("--tta-seed", type=int, default=42)

    parser.add_argument(
        "--comparison-mode",
        choices=["whole", "pairwise", "adaptive"],
        default="whole",
        help=(
            "whole: single VLM call orders all 4 images at once (default, with "
            "permutation TTA). pairwise: 6 two-image comparisons per sample, "
            "combined via best-agreement search over the 24 valid orders. "
            "adaptive: whole-mode TTA first, with one extra targeted pairwise "
            "verification call only when the top-2 TTA candidates disagree by "
            "exactly one swapped pair."
        ),
    )
    parser.add_argument(
        "--pairwise-max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens per two-image comparison call in pairwise mode.",
    )
    parser.add_argument(
        "--pairwise-symmetry-check",
        action="store_true",
        help=(
            "In pairwise mode, ask each pair twice with swapped image presentation "
            "order and keep only judgments that agree both times (doubles pairwise "
            "VLM calls from 6 to 12 per sample)."
        ),
    )

    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--qwen-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()


def resolve_captions_for_row(row, image_dir: Path, args: argparse.Namespace, captioner):
    return generate_fresh_captions_for_row(
        row=row,
        image_dir=image_dir,
        captioner=captioner,
        caption_max_new_tokens=args.caption_max_new_tokens,
        max_caption_chars=args.max_caption_chars,
        sentence_aware=args.sentence_aware_captions,
    )


def _collect_tta_orders(
    row,
    captions: list[str],
    image_dir: Path,
    orderer: QwenOrderer,
    tta_permutations: list[list[int]],
    order_max_new_tokens: int,
) -> tuple[list[list[int]], list[dict[str, object]]]:
    """Run one whole-order VLM call per TTA permutation and restore each result
    to original image numbering. Shared by whole mode and adaptive mode so both
    read the same TTA votes rather than duplicating the generation loop."""
    valid_orders: list[list[int]] = []
    tta_outputs: list[dict[str, object]] = []
    for permutation in tta_permutations:
        permuted_row, permuted_captions = permute_row_and_captions(row, captions, permutation)
        messages = build_order_messages(permuted_row, image_dir=image_dir, captions=permuted_captions)
        output_text = orderer.generate_order(messages, max_new_tokens=order_max_new_tokens)

        try:
            permuted_order = parse_permutation_from_text(output_text)
            restored_order = restore_chronological_order(permuted_order, permutation)
            valid_orders.append(restored_order)
        except ValueError:
            permuted_order = None
            restored_order = None

        tta_outputs.append(
            {
                "permutation_new_slot_to_original": permutation,
                "model_output": output_text,
                "parsed_permuted_chronological_order": permuted_order,
                "restored_chronological_order": restored_order,
            }
        )
    return valid_orders, tta_outputs


def run_whole_order_inference(
    row,
    captions: list[str],
    image_dir: Path,
    orderer: QwenOrderer,
    tta_permutations: list[list[int]],
    order_max_new_tokens: int,
    fallback: list[int],
) -> tuple[list[int], dict[str, object]]:
    """Default single-shot listwise ordering: one VLM call per TTA permutation."""
    valid_orders, tta_outputs = _collect_tta_orders(
        row, captions, image_dir, orderer, tta_permutations, order_max_new_tokens
    )

    if valid_orders:
        chronological_order, winning_votes = consensus_chronological_order(valid_orders)
        pred_list = chronological_to_submission(chronological_order)
        used_fallback = False
    else:
        chronological_order = None
        winning_votes = 0
        pred_list = fallback
        used_fallback = True

    vote_counts = Counter(tuple(order) for order in valid_orders)
    raw_record = {
        "comparison_mode": "whole",
        "tta_outputs": tta_outputs,
        "tta_valid_prediction_count": len(valid_orders),
        "tta_vote_counts": [
            {"chronological_order": list(order), "votes": votes} for order, votes in vote_counts.most_common()
        ],
        "consensus_chronological_order": chronological_order,
        "consensus_votes": winning_votes,
        "parsed_submission_answer": pred_list,
        "used_fallback": used_fallback,
    }
    return pred_list, raw_record


def run_pairwise_order_inference(
    row,
    captions: list[str],
    image_dir: Path,
    orderer: QwenOrderer,
    pairwise_max_new_tokens: int,
    pairwise_symmetry_check: bool,
    fallback: list[int],
) -> tuple[list[int], dict[str, object]]:
    """Compare images two at a time (6 pairs) and combine judgments by searching
    the 24 valid permutations for the one with the highest pairwise agreement."""
    judgments, pairwise_outputs = collect_pairwise_judgments(
        row,
        image_dir,
        captions,
        orderer,
        max_new_tokens=pairwise_max_new_tokens,
        symmetry_check=pairwise_symmetry_check,
    )
    chronological_order, agreement, judged_pair_count = best_order_from_pairwise(judgments)

    if judged_pair_count > 0:
        pred_list = chronological_to_submission(chronological_order)
        used_fallback = False
    else:
        chronological_order = None
        pred_list = fallback
        used_fallback = True

    raw_record = {
        "comparison_mode": "pairwise",
        "pairwise_outputs": pairwise_outputs,
        "judged_pair_count": judged_pair_count,
        "pairwise_agreement_count": agreement,
        "consensus_chronological_order": chronological_order,
        "parsed_submission_answer": pred_list,
        "used_fallback": used_fallback,
    }
    return pred_list, raw_record


def run_adaptive_order_inference(
    row,
    captions: list[str],
    image_dir: Path,
    orderer: QwenOrderer,
    tta_permutations: list[list[int]],
    order_max_new_tokens: int,
    pairwise_max_new_tokens: int,
    fallback: list[int],
) -> tuple[list[int], dict[str, object]]:
    """Whole-order TTA first; only escalate to a single targeted pairwise
    verification call when the top-2 TTA candidates disagree by exactly one
    swapped pair. This keeps the common case at whole-mode's cost (one VLM
    call per TTA view) and spends at most one extra call on genuinely
    disputed samples, rather than always running all 6 pairwise comparisons.

    Uses the same Qwen checkpoint/adapter and prompt-only branching as whole
    and pairwise mode (no second model, so this isn't model ensembling).
    """
    valid_orders, tta_outputs = _collect_tta_orders(
        row, captions, image_dir, orderer, tta_permutations, order_max_new_tokens
    )
    vote_counts = Counter(tuple(order) for order in valid_orders)
    ranked = vote_counts.most_common()

    disputed_pair: tuple[int, int] | None = None
    pairwise_verification: dict[str, object] | None = None

    if not ranked:
        chronological_order = None
        pred_list = fallback
        used_fallback = True
    elif len(ranked) == 1:
        # All TTA views agree: confident, skip the extra pairwise call entirely.
        chronological_order = list(ranked[0][0])
        pred_list = chronological_to_submission(chronological_order)
        used_fallback = False
    else:
        top1_order, _top1_votes = ranked[0]
        top2_order, _top2_votes = ranked[1]
        disputed_pair = find_disputed_pair(top1_order, top2_order)

        if disputed_pair is None:
            # Top-2 candidates disagree by more than one swap; one pairwise
            # call can't adjudicate that, so keep the higher-vote candidate.
            chronological_order = list(top1_order)
        else:
            messages = build_pairwise_messages(
                row, image_dir, captions, disputed_pair, include_story_context=True
            )
            output_text = orderer.generate_order(messages, max_new_tokens=pairwise_max_new_tokens)
            judged_order = parse_pair_order_from_text(output_text, disputed_pair)
            pairwise_verification = {
                "pair": list(disputed_pair),
                "model_output": output_text,
                "parsed_order": list(judged_order) if judged_order is not None else None,
            }
            chronological_order = resolve_disputed_pair(top1_order, top2_order, disputed_pair, judged_order)

        pred_list = chronological_to_submission(chronological_order)
        used_fallback = False

    vote_counts_serialized = [
        {"chronological_order": list(order), "votes": votes} for order, votes in ranked
    ]
    raw_record = {
        "comparison_mode": "adaptive",
        "tta_outputs": tta_outputs,
        "tta_valid_prediction_count": len(valid_orders),
        "tta_vote_counts": vote_counts_serialized,
        "disputed_pair": list(disputed_pair) if disputed_pair is not None else None,
        "pairwise_verification": pairwise_verification,
        "consensus_chronological_order": chronological_order,
        "parsed_submission_answer": pred_list,
        "used_fallback": used_fallback,
    }
    return pred_list, raw_record


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    test_df = read_csv(data_dir / "test.csv")
    sample_df = read_csv(data_dir / "sample_submission.csv")
    if args.max_samples is not None:
        test_df = test_df.head(args.max_samples).copy()
        sample_df = sample_df.head(args.max_samples).copy()

    fallback = normalize_permutation(ast.literal_eval(args.fallback_answer))
    tta_permutations = build_tta_permutations(args.tta_permutations, seed=args.tta_seed)
    image_dir = data_dir / "test"

    if args.caption_cache:
        print("Ignoring --caption-cache during inference; captions are generated fresh for every sample.")
    print(f"Loading caption backend: {args.caption_backend}")
    captioner = build_captioner(args)

    print(f"Loading ordering model: {args.order_model}")
    orderer = QwenOrderer(
        model_name=args.order_model,
        device_map=args.device_map,
        torch_dtype=args.qwen_torch_dtype,
        attn_implementation=args.attn_implementation,
        adapter_path=args.order_adapter,
    )

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, str]] = []
    if args.comparison_mode == "pairwise":
        mode_note = "in pairwise mode (6 two-image comparisons per sample)"
    elif args.comparison_mode == "adaptive":
        mode_note = (
            f"in adaptive mode ({len(tta_permutations)} whole-order TTA view(s), "
            "+1 pairwise verification call only on disputed samples)"
        )
    else:
        mode_note = f"in whole mode with {len(tta_permutations)} permutation view(s)"
    print(f"Running fresh-caption inference on {len(test_df)} samples {mode_note}")
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            captions = resolve_captions_for_row(
                row=row,
                image_dir=image_dir,
                args=args,
                captioner=captioner,
            )

            if args.comparison_mode == "pairwise":
                pred_list, raw_record = run_pairwise_order_inference(
                    row,
                    captions,
                    image_dir,
                    orderer,
                    pairwise_max_new_tokens=args.pairwise_max_new_tokens,
                    pairwise_symmetry_check=args.pairwise_symmetry_check,
                    fallback=fallback,
                )
            elif args.comparison_mode == "adaptive":
                pred_list, raw_record = run_adaptive_order_inference(
                    row,
                    captions,
                    image_dir,
                    orderer,
                    tta_permutations,
                    order_max_new_tokens=args.order_max_new_tokens,
                    pairwise_max_new_tokens=args.pairwise_max_new_tokens,
                    fallback=fallback,
                )
            else:
                pred_list, raw_record = run_whole_order_inference(
                    row,
                    captions,
                    image_dir,
                    orderer,
                    tta_permutations,
                    order_max_new_tokens=args.order_max_new_tokens,
                    fallback=fallback,
                )

            predictions.append({"Id": row["Id"], "Answer": format_answer(pred_list)})
            raw_file.write(
                json.dumps({"Id": row["Id"], "captions": captions, **raw_record}, ensure_ascii=False) + "\n"
            )

    write_submission(predictions, args.output, sample_df=sample_df)
    print(f"Saved submission to {args.output}")
    print(f"Saved raw outputs to {raw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
