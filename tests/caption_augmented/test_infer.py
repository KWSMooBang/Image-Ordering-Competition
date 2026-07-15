from pathlib import Path

import pandas as pd

from src.caption_augmented.infer import run_adaptive_order_inference
from src.caption_augmented.tta import build_tta_permutations
from src.submission import chronological_to_submission

CAPTIONS = ["first", "second", "third", "fourth"]
FALLBACK = [1, 2, 3, 4]


def make_row() -> pd.Series:
    return pd.Series(
        {
            "Id": "sample-1",
            "Input_1": "a.jpg",
            "Input_2": "b.jpg",
            "Input_3": "c.jpg",
            "Input_4": "d.jpg",
            "Sentence": "A person opens a box and takes out a cup.",
        }
    )


class ScriptedOrderer:
    """Fake orderer returning a scripted response per call, in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def generate_order(self, messages, max_new_tokens):
        self.calls.append(messages)
        return self._responses.pop(0)


TTA_PERMUTATIONS = build_tta_permutations(2, seed=42)  # [[1,2,3,4], [2,3,4,1]]


def test_adaptive_unanimous_tta_skips_pairwise_call():
    # Both TTA views restore to the same chronological order -> confident, no
    # pairwise verification call should be made.
    responses = ["[3, 1, 4, 2]", "[3, 1, 4, 2]"]
    orderer = ScriptedOrderer(responses)

    pred_list, raw_record = run_adaptive_order_inference(
        make_row(),
        CAPTIONS,
        image_dir=Path("/data/test"),
        orderer=orderer,
        tta_permutations=TTA_PERMUTATIONS,
        order_max_new_tokens=64,
        pairwise_max_new_tokens=32,
        fallback=FALLBACK,
    )

    assert len(orderer.calls) == 2  # no third (pairwise) call
    assert raw_record["disputed_pair"] is None
    assert raw_record["pairwise_verification"] is None
    assert raw_record["consensus_chronological_order"] == [3, 1, 4, 2]
    assert pred_list == chronological_to_submission([3, 1, 4, 2])


def test_adaptive_single_swap_dispute_triggers_one_verification_call_and_can_agree():
    # View1 (identity permutation) restores to [3,1,4,2].
    # View2 (permutation [2,3,4,1]) restores to [1,3,4,2] - a single swap of
    # images 1 and 3 relative to view1. See module docstring math in the PR:
    # raw output "[4, 2, 3, 1]" under permutation [2,3,4,1] restores to [1,3,4,2].
    responses = ["[3, 1, 4, 2]", "[4, 2, 3, 1]", "[3, 1]"]  # 3rd = pairwise verification, agrees with top1
    orderer = ScriptedOrderer(responses)

    pred_list, raw_record = run_adaptive_order_inference(
        make_row(),
        CAPTIONS,
        image_dir=Path("/data/test"),
        orderer=orderer,
        tta_permutations=TTA_PERMUTATIONS,
        order_max_new_tokens=64,
        pairwise_max_new_tokens=32,
        fallback=FALLBACK,
    )

    assert len(orderer.calls) == 3  # exactly one extra pairwise verification call
    assert raw_record["disputed_pair"] == [1, 3]
    assert raw_record["pairwise_verification"]["parsed_order"] == [3, 1]
    # Judgment (3 before 1) agrees with top1's arrangement -> keep top1.
    assert raw_record["consensus_chronological_order"] == [3, 1, 4, 2]
    assert pred_list == chronological_to_submission([3, 1, 4, 2])


def test_adaptive_single_swap_dispute_can_switch_to_second_candidate():
    responses = ["[3, 1, 4, 2]", "[4, 2, 3, 1]", "[1, 3]"]  # verification disagrees with top1
    orderer = ScriptedOrderer(responses)

    pred_list, raw_record = run_adaptive_order_inference(
        make_row(),
        CAPTIONS,
        image_dir=Path("/data/test"),
        orderer=orderer,
        tta_permutations=TTA_PERMUTATIONS,
        order_max_new_tokens=64,
        pairwise_max_new_tokens=32,
        fallback=FALLBACK,
    )

    assert len(orderer.calls) == 3
    # Judgment (1 before 3) disagrees with top1 -> switch to the second candidate.
    assert raw_record["consensus_chronological_order"] == [1, 3, 4, 2]
    assert pred_list == chronological_to_submission([1, 3, 4, 2])


def test_adaptive_multi_position_disagreement_keeps_top1_without_verification_call():
    # View1 restores to [3,1,4,2]; make view2 restore to something 3+ positions
    # different (a 3-cycle at least) so a single pairwise call can't resolve it.
    # Under permutation [2,3,4,1], raw "[1, 2, 3, 4]" restores to [2,3,4,1].
    responses = ["[3, 1, 4, 2]", "[1, 2, 3, 4]"]
    orderer = ScriptedOrderer(responses)

    pred_list, raw_record = run_adaptive_order_inference(
        make_row(),
        CAPTIONS,
        image_dir=Path("/data/test"),
        orderer=orderer,
        tta_permutations=TTA_PERMUTATIONS,
        order_max_new_tokens=64,
        pairwise_max_new_tokens=32,
        fallback=FALLBACK,
    )

    assert len(orderer.calls) == 2  # no verification call attempted
    assert raw_record["disputed_pair"] is None
    assert raw_record["pairwise_verification"] is None
    assert raw_record["consensus_chronological_order"] == [3, 1, 4, 2]  # kept top1
    assert pred_list == chronological_to_submission([3, 1, 4, 2])


def test_adaptive_falls_back_when_no_tta_view_parses():
    responses = ["not a list", "still not a list"]
    orderer = ScriptedOrderer(responses)

    pred_list, raw_record = run_adaptive_order_inference(
        make_row(),
        CAPTIONS,
        image_dir=Path("/data/test"),
        orderer=orderer,
        tta_permutations=TTA_PERMUTATIONS,
        order_max_new_tokens=64,
        pairwise_max_new_tokens=32,
        fallback=FALLBACK,
    )

    assert len(orderer.calls) == 2
    assert raw_record["used_fallback"] is True
    assert raw_record["consensus_chronological_order"] is None
    assert pred_list == FALLBACK
