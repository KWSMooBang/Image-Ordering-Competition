from __future__ import annotations

import ast
import re
from itertools import permutations
from pathlib import Path
from typing import Any, Protocol, Sequence

import pandas as pd

from src.data_utils import INPUT_COLUMNS
from src.submission import PERMUTATION, normalize_permutation

# The six unordered image pairs among four frames.
ALL_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

# All 24 valid chronological orders, used as the search space when combining
# pairwise judgments into one final order (there is no separate "pairwise
# ranking model": this brute-force search over 24 candidates plays that role).
ALL_ORDER_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(permutations(PERMUTATION))

_LIST_PATTERN = re.compile(r"\[[^\[\]]+\]")


class PairwiseOrderer(Protocol):
    def generate_order(self, messages: list[dict[str, object]], max_new_tokens: int) -> str: ...


def build_pairwise_messages(
    row: pd.Series,
    image_dir: Path,
    captions: Sequence[str],
    pair: tuple[int, int],
    *,
    include_story_context: bool = False,
) -> list[dict[str, Any]]:
    """Build a two-image chat prompt asking which image of `pair` happens first.

    Unlike `build_order_messages`, only the two images in `pair` are shown to the
    model. `pair` also fixes presentation order (first element shown first).

    With `include_story_context=True`, all four captions are listed as a
    "full sequence" reference block even though only the two images in `pair`
    are shown. This is meant for adaptive verification (a targeted recheck of
    one disputed pair from an otherwise-confident whole-order prediction),
    where the extra context helps without reintroducing all four images. The
    default (False) preserves the plain two-image-only prompt used by the
    standalone pairwise comparison mode.
    """
    if len(captions) != 4:
        raise ValueError(f"Expected 4 captions, got {len(captions)}")

    image_a, image_b = pair
    content: list[dict[str, str]] = []

    if include_story_context:
        caption_lines = "\n".join(f"{index}. {caption}" for index, caption in enumerate(captions, start=1))
        content.append(
            {
                "type": "text",
                "text": (
                    f'Storyline: "{row["Sentence"]}"\n\n'
                    f"Full sequence captions (for context only, not all shown as images):\n{caption_lines}\n"
                ),
            }
        )

    for image_index in (image_a, image_b):
        column = INPUT_COLUMNS[image_index - 1]
        image_path = image_dir / str(row["Id"]) / str(row[column])
        content.append({"type": "image", "image": str(image_path)})
        content.append(
            {"type": "text", "text": f"\nImage {image_index} caption: {captions[image_index - 1]}\n"}
        )

    story_line = "" if include_story_context else f'Story sentence: "{row["Sentence"]}"\n'
    content.append(
        {
            "type": "text",
            "text": (
                f"{story_line}"
                "The two captions above were generated automatically and may be imperfect. "
                "Use the images as primary evidence and the captions as supporting notes. "
                f"Consider only Image {image_a} and Image {image_b} and decide which one happens "
                "first chronologically in the story. "
                "Return ONLY a Python list containing both image labels ordered chronologically, "
                f"earliest first. Example: [{image_b}, {image_a}]"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def parse_pair_order_from_text(
    text: str,
    pair: tuple[int, int],
    fallback: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Parse a two-item chronological order for `pair` from free-form model text.

    Returns None (rather than raising) when no valid two-item permutation of
    `pair` can be found and no fallback is given, since a single unparsable pair
    judgment should not abort the whole sample.
    """
    valid = set(pair)
    for match in _LIST_PATTERN.finditer(text):
        try:
            parsed = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        try:
            values = [int(value) for value in parsed]
        except (TypeError, ValueError):
            continue
        if len(values) == 2 and set(values) == valid and values[0] != values[1]:
            return (values[0], values[1])

    return fallback


def score_permutation_against_pairs(
    order: Sequence[int],
    pair_orders: dict[tuple[int, int], tuple[int, int]],
) -> int:
    """Count how many pairwise judgments a full chronological order agrees with."""
    position = {image_number: index for index, image_number in enumerate(order)}
    agreement = 0
    for earlier, later in pair_orders.values():
        if position[earlier] < position[later]:
            agreement += 1
    return agreement


def best_order_from_pairwise(
    pair_orders: dict[tuple[int, int], tuple[int, int]],
) -> tuple[list[int], int, int]:
    """Search all 24 permutations for the one most consistent with pairwise judgments.

    This is how cyclic/contradictory judgments (e.g. 1<2, 2<3, 3<1) are resolved:
    there is no permutation that satisfies every pair, so the permutation with the
    highest agreement count wins. Ties are broken by `ALL_ORDER_PERMUTATIONS` order
    (starting with the identity permutation), which keeps the choice deterministic.

    Returns `(best_order, agreement_count, judged_pair_count)`. With zero judged
    pairs, returns the identity order with agreement 0 so callers can detect this
    case via `judged_pair_count == 0` and fall back explicitly if they prefer.
    """
    judged_pair_count = len(pair_orders)
    if judged_pair_count == 0:
        return list(PERMUTATION), 0, 0

    best_order = ALL_ORDER_PERMUTATIONS[0]
    best_score = -1
    for candidate in ALL_ORDER_PERMUTATIONS:
        score = score_permutation_against_pairs(candidate, pair_orders)
        if score > best_score:
            best_score = score
            best_order = candidate
    return list(best_order), best_score, judged_pair_count


def find_disputed_pair(order_a: Sequence[int], order_b: Sequence[int]) -> tuple[int, int] | None:
    """Return the two image numbers swapped between `order_a` and `order_b`.

    Returns `None` when the two orders are identical, or when they differ by
    more than a single transposition (3+ positions differ). A single targeted
    pairwise re-check can only resolve a disagreement caused by exactly one
    swapped pair; anything more tangled is left to `order_a` (the higher-vote
    candidate) rather than guessed at.
    """
    order_a = list(order_a)
    order_b = list(order_b)
    if order_a == order_b:
        return None

    diff_positions = [index for index in range(len(order_a)) if order_a[index] != order_b[index]]
    if len(diff_positions) != 2:
        return None

    first, second = diff_positions
    if order_a[first] != order_b[second] or order_a[second] != order_b[first]:
        return None  # not a pure transposition; shouldn't happen for two valid permutations of the same set

    return tuple(sorted((order_a[first], order_a[second])))


def resolve_disputed_pair(
    candidate_a: Sequence[int],
    candidate_b: Sequence[int],
    disputed_pair: tuple[int, int],
    judged_order: tuple[int, int] | None,
) -> list[int]:
    """Pick between two single-swap-apart candidates using one pairwise judgment.

    `candidate_a` is treated as the default (e.g. the higher-vote whole-order
    candidate). If `judged_order` is `None` (unparsable verification response)
    or agrees with how `candidate_a` orders `disputed_pair`, `candidate_a` is
    returned; otherwise `candidate_b` is returned, since by construction it is
    exactly `candidate_a` with `disputed_pair` swapped.
    """
    if judged_order is None:
        return list(candidate_a)

    position_in_a = {image_number: index for index, image_number in enumerate(candidate_a)}
    image_x, image_y = disputed_pair
    a_earlier, a_later = (image_x, image_y) if position_in_a[image_x] < position_in_a[image_y] else (image_y, image_x)

    if judged_order == (a_earlier, a_later):
        return list(candidate_a)
    return list(candidate_b)



def collect_pairwise_judgments(
    row: pd.Series,
    image_dir: Path,
    captions: Sequence[str],
    orderer: PairwiseOrderer,
    *,
    max_new_tokens: int = 32,
    symmetry_check: bool = False,
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[dict[str, Any]]]:
    """Query `orderer` once (or twice, with `symmetry_check`) per image pair.

    With `symmetry_check`, each pair is asked twice with the two images shown in
    swapped presentation order; a judgment is only kept if both queries agree,
    which screens out judgments driven by presentation-order bias rather than
    actual visual/textual evidence. A pair with no resolvable judgment is simply
    left out of the returned mapping; `best_order_from_pairwise` handles the
    remaining pairs.
    """
    judgments: dict[tuple[int, int], tuple[int, int]] = {}
    raw_outputs: list[dict[str, Any]] = []

    for pair in ALL_PAIRS:
        messages = build_pairwise_messages(row, image_dir, captions, pair)
        output_text = orderer.generate_order(messages, max_new_tokens=max_new_tokens)
        judged = parse_pair_order_from_text(output_text, pair)

        record: dict[str, Any] = {
            "pair": list(pair),
            "model_output": output_text,
            "parsed_order": list(judged) if judged is not None else None,
        }

        if symmetry_check:
            swapped_pair = (pair[1], pair[0])
            swapped_messages = build_pairwise_messages(row, image_dir, captions, swapped_pair)
            swapped_output_text = orderer.generate_order(swapped_messages, max_new_tokens=max_new_tokens)
            swapped_judged = parse_pair_order_from_text(swapped_output_text, swapped_pair)

            record["swapped_model_output"] = swapped_output_text
            record["swapped_parsed_order"] = list(swapped_judged) if swapped_judged is not None else None

            if judged is not None and judged == swapped_judged:
                judgments[pair] = judged
            record["symmetry_agreed"] = judged is not None and judged == swapped_judged
        elif judged is not None:
            judgments[pair] = judged

        raw_outputs.append(record)

    return judgments, raw_outputs
