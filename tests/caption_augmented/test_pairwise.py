from pathlib import Path

import pandas as pd
import pytest

from src.caption_augmented.pairwise import (
    ALL_PAIRS,
    best_order_from_pairwise,
    build_pairwise_messages,
    collect_pairwise_judgments,
    find_disputed_pair,
    parse_pair_order_from_text,
    resolve_disputed_pair,
    score_permutation_against_pairs,
)


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


CAPTIONS = ["first", "second", "third", "fourth"]


class ScriptedOrderer:
    """Fake orderer returning a scripted response per call, in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def generate_order(self, messages, max_new_tokens):
        self.calls.append(messages)
        return self._responses.pop(0)


def pair_orders_from_full_order(order, pairs=ALL_PAIRS):
    position = {image_number: index for index, image_number in enumerate(order)}
    return {
        pair: (pair[0], pair[1]) if position[pair[0]] < position[pair[1]] else (pair[1], pair[0])
        for pair in pairs
    }


# ---------------------------------------------------------------------------
# build_pairwise_messages
# ---------------------------------------------------------------------------


def test_build_pairwise_messages_only_shows_the_two_images_in_pair():
    row = make_row()
    image_dir = Path("/data/test")
    messages = build_pairwise_messages(row, image_dir=image_dir, captions=CAPTIONS, pair=(2, 4))
    content = messages[0]["content"]

    image_items = [item for item in content if item["type"] == "image"]
    text = "\n".join(item["text"] for item in content if item["type"] == "text")

    assert [item["image"] for item in image_items] == [
        str(image_dir / "sample-1" / "b.jpg"),
        str(image_dir / "sample-1" / "d.jpg"),
    ]
    assert "Image 2 caption: second" in text
    assert "Image 4 caption: fourth" in text
    assert "Consider only Image 2 and Image 4" in text
    # Images not in the pair must not leak into the prompt.
    assert "Image 1 caption" not in text
    assert "Image 3 caption" not in text


def test_build_pairwise_messages_requires_four_captions():
    with pytest.raises(ValueError, match="Expected 4 captions"):
        build_pairwise_messages(make_row(), image_dir=Path("/data/test"), captions=["only one"], pair=(1, 2))


def test_build_pairwise_messages_with_story_context_lists_all_four_captions():
    row = make_row()
    image_dir = Path("/data/test")
    messages = build_pairwise_messages(
        row,
        image_dir=image_dir,
        captions=CAPTIONS,
        pair=(2, 4),
        include_story_context=True,
    )
    content = messages[0]["content"]
    text = "\n".join(item["text"] for item in content if item["type"] == "text")
    image_items = [item for item in content if item["type"] == "image"]

    # Still only the two disputed images are actually shown.
    assert [item["image"] for item in image_items] == [
        str(image_dir / "sample-1" / "b.jpg"),
        str(image_dir / "sample-1" / "d.jpg"),
    ]
    # But all four captions appear as context text, unlike the default prompt.
    assert "1. first" in text
    assert "2. second" in text
    assert "3. third" in text
    assert "4. fourth" in text
    assert "Consider only Image 2 and Image 4" in text


def test_build_pairwise_messages_without_story_context_omits_other_captions():
    messages = build_pairwise_messages(
        make_row(), image_dir=Path("/data/test"), captions=CAPTIONS, pair=(2, 4)
    )
    text = "\n".join(item["text"] for item in messages[0]["content"] if item["type"] == "text")
    assert "Image 1 caption" not in text
    assert "Image 3 caption" not in text


# ---------------------------------------------------------------------------
# parse_pair_order_from_text
# ---------------------------------------------------------------------------


def test_parse_pair_order_from_text_extracts_matching_list():
    assert parse_pair_order_from_text("Sure, [2, 1] seems right.", pair=(1, 2)) == (2, 1)


def test_parse_pair_order_from_text_ignores_unrelated_lists():
    text = "Here is some other list [9, 9] but the answer is [4, 2]."
    assert parse_pair_order_from_text(text, pair=(2, 4)) == (4, 2)


def test_parse_pair_order_from_text_falls_back_when_unparsable():
    assert parse_pair_order_from_text("I cannot tell.", pair=(1, 2), fallback=(1, 2)) == (1, 2)
    assert parse_pair_order_from_text("I cannot tell.", pair=(1, 2)) is None


# ---------------------------------------------------------------------------
# score_permutation_against_pairs / best_order_from_pairwise
# ---------------------------------------------------------------------------


def test_best_order_from_pairwise_recovers_a_fully_consistent_order():
    true_order = [3, 1, 4, 2]
    pair_orders = pair_orders_from_full_order(true_order)

    order, agreement, judged_pair_count = best_order_from_pairwise(pair_orders)

    assert order == true_order
    assert agreement == 6
    assert judged_pair_count == 6


def test_best_order_from_pairwise_handles_cyclic_judgments_deterministically():
    # 1<2, 2<3, 3<1 is a cycle: no permutation can satisfy all three, so the
    # best achievable agreement is 2 out of 3.
    pair_orders = {(1, 2): (1, 2), (2, 3): (2, 3), (1, 3): (3, 1)}

    order, agreement, judged_pair_count = best_order_from_pairwise(pair_orders)

    assert agreement == 2
    assert judged_pair_count == 3
    assert score_permutation_against_pairs(order, pair_orders) == 2


def test_best_order_from_pairwise_with_no_judgments_returns_identity_and_zero_score():
    order, agreement, judged_pair_count = best_order_from_pairwise({})

    assert order == [1, 2, 3, 4]
    assert agreement == 0
    assert judged_pair_count == 0


# ---------------------------------------------------------------------------
# collect_pairwise_judgments
# ---------------------------------------------------------------------------


def test_collect_pairwise_judgments_without_symmetry_check_makes_one_call_per_pair():
    responses = ["[2, 1]", "[3, 1]", "[4, 1]", "[3, 2]", "[4, 2]", "[4, 3]"]
    orderer = ScriptedOrderer(responses)

    judgments, raw_outputs = collect_pairwise_judgments(
        make_row(), Path("/data/test"), CAPTIONS, orderer, max_new_tokens=32
    )

    assert len(orderer.calls) == 6
    assert len(raw_outputs) == 6
    assert judgments == {
        (1, 2): (2, 1),
        (1, 3): (3, 1),
        (1, 4): (4, 1),
        (2, 3): (3, 2),
        (2, 4): (4, 2),
        (3, 4): (4, 3),
    }
    order, agreement, judged_pair_count = best_order_from_pairwise(judgments)
    assert order == [4, 3, 2, 1]
    assert (agreement, judged_pair_count) == (6, 6)


def test_collect_pairwise_judgments_with_symmetry_check_drops_disagreements():
    # One call pair per ALL_PAIRS entry when symmetry_check is on, in order:
    # (pair as-is, swapped pair). Only (1, 3) disagrees between the two calls.
    responses = [
        "[2, 1]", "[2, 1]",  # (1, 2) -> agree
        "[3, 1]", "[1, 3]",  # (1, 3) -> disagree
        "[4, 1]", "[4, 1]",  # (1, 4) -> agree
        "[3, 2]", "[3, 2]",  # (2, 3) -> agree
        "[4, 2]", "[4, 2]",  # (2, 4) -> agree
        "[4, 3]", "[4, 3]",  # (3, 4) -> agree
    ]
    orderer = ScriptedOrderer(responses)

    judgments, raw_outputs = collect_pairwise_judgments(
        make_row(),
        Path("/data/test"),
        CAPTIONS,
        orderer,
        max_new_tokens=32,
        symmetry_check=True,
    )

    assert len(orderer.calls) == 12
    assert len(raw_outputs) == 6
    assert (1, 3) not in judgments
    assert judgments == {
        (1, 2): (2, 1),
        (1, 4): (4, 1),
        (2, 3): (3, 2),
        (2, 4): (4, 2),
        (3, 4): (4, 3),
    }
    disagreeing_record = next(record for record in raw_outputs if tuple(record["pair"]) == (1, 3))
    assert disagreeing_record["symmetry_agreed"] is False


# ---------------------------------------------------------------------------
# find_disputed_pair / resolve_disputed_pair (adaptive whole+pairwise cascade)
# ---------------------------------------------------------------------------


def test_find_disputed_pair_identical_orders_returns_none():
    assert find_disputed_pair([1, 2, 3, 4], [1, 2, 3, 4]) is None


def test_find_disputed_pair_single_swap_returns_the_swapped_pair():
    # Positions 1 and 2 (0-indexed) hold 3 and 4 in one order, swapped in the other.
    assert find_disputed_pair([1, 3, 4, 2], [1, 4, 3, 2]) == (3, 4)


def test_find_disputed_pair_more_than_one_swap_returns_none():
    # A 3-cycle: three positions differ, which one pairwise recheck cannot resolve.
    assert find_disputed_pair([1, 2, 3, 4], [1, 3, 4, 2]) is None
    # All four positions differ.
    assert find_disputed_pair([1, 2, 3, 4], [4, 3, 2, 1]) is None


def test_resolve_disputed_pair_keeps_candidate_a_when_judgment_agrees():
    candidate_a = [1, 3, 4, 2]
    candidate_b = [1, 4, 3, 2]
    # candidate_a places 3 before 4.
    resolved = resolve_disputed_pair(candidate_a, candidate_b, disputed_pair=(3, 4), judged_order=(3, 4))
    assert resolved == candidate_a


def test_resolve_disputed_pair_switches_to_candidate_b_when_judgment_disagrees():
    candidate_a = [1, 3, 4, 2]
    candidate_b = [1, 4, 3, 2]
    # Judgment says 4 comes before 3, contradicting candidate_a.
    resolved = resolve_disputed_pair(candidate_a, candidate_b, disputed_pair=(3, 4), judged_order=(4, 3))
    assert resolved == candidate_b


def test_resolve_disputed_pair_falls_back_to_candidate_a_when_unparsable():
    candidate_a = [1, 3, 4, 2]
    candidate_b = [1, 4, 3, 2]
    resolved = resolve_disputed_pair(candidate_a, candidate_b, disputed_pair=(3, 4), judged_order=None)
    assert resolved == candidate_a