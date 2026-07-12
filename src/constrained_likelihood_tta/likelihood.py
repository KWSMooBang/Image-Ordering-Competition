from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

from src.submission import PERMUTATION, normalize_permutation

CandidateOrder = tuple[int, int, int, int]
ALL_CANDIDATE_ORDERS: tuple[CandidateOrder, ...] = tuple(
    tuple(order) for order in permutations(PERMUTATION)
)


@dataclass(frozen=True)
class CandidateLikelihood:
    order: CandidateOrder
    log_likelihood: float
    token_count: int
    score: float


def as_candidate_order(order: Sequence[int]) -> CandidateOrder:
    return tuple(normalize_permutation(order))  # type: ignore[return-value]


def candidate_text(order: Sequence[int]) -> str:
    return str(list(as_candidate_order(order)))


def chunk_candidates(
    candidates: Sequence[CandidateOrder],
    batch_size: int,
) -> list[list[CandidateOrder]]:
    if batch_size < 2 or batch_size > len(PERMUTATION):
        raise ValueError("candidate batch size must be between 2 and 4")
    if len(candidates) < 2:
        raise ValueError("at least two candidate orders are required")

    buckets = {
        first_image: [
            as_candidate_order(order)
            for order in candidates
            if int(order[0]) == first_image
        ]
        for first_image in PERMUTATION
    }
    chunks: list[list[CandidateOrder]] = []
    cursor = 0
    while any(buckets.values()):
        chunk: list[CandidateOrder] = []
        for offset in range(len(PERMUTATION)):
            first_image = PERMUTATION[(cursor + offset) % len(PERMUTATION)]
            if buckets[first_image]:
                chunk.append(buckets[first_image].pop(0))
            if len(chunk) == batch_size:
                break
        if len(chunk) < 2:
            raise ValueError(
                "candidate groups require at least two distinct first-image labels"
            )
        if len({order[0] for order in chunk}) != len(chunk):
            raise RuntimeError("candidate group first-image labels must be unique")
        chunks.append(chunk)
        cursor = (cursor + batch_size) % len(PERMUTATION)

    if sum(len(chunk) for chunk in chunks) != len(candidates):
        raise RuntimeError("candidate grouping lost one or more orders")
    return chunks


class CandidateTokenConstraint:
    """Prefix constraint permitting only a fixed set of token sequences."""

    def __init__(
        self,
        token_sequences: Sequence[Sequence[int]],
        *,
        prompt_length: int,
    ) -> None:
        sequences = [
            tuple(int(token) for token in sequence) for sequence in token_sequences
        ]
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("candidate token sequences must be non-empty")
        if len(set(sequences)) != len(sequences):
            raise ValueError("candidate token sequences must be unique")
        self.sequences = tuple(sequences)
        self.prompt_length = int(prompt_length)

    def allowed_next_tokens(self, input_ids: Sequence[int]) -> list[int]:
        generated = tuple(int(token) for token in input_ids[self.prompt_length :])
        allowed = {
            sequence[len(generated)]
            for sequence in self.sequences
            if len(generated) < len(sequence)
            and sequence[: len(generated)] == generated
        }
        if not allowed:
            raise RuntimeError(f"Prefix is outside the candidate trie: {generated}")
        return sorted(allowed)

    def __call__(self, _batch_id: int, input_ids) -> list[int]:
        values = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        return self.allowed_next_tokens(values)


def restore_candidate_scores(
    scores: Mapping[Sequence[int], float],
    permutation: Sequence[int],
) -> dict[CandidateOrder, float]:
    new_slot_to_original = normalize_permutation(permutation)
    restored: dict[CandidateOrder, float] = {}
    for order, score in scores.items():
        permuted_order = normalize_permutation(order)
        original_order = as_candidate_order(
            [new_slot_to_original[new_slot - 1] for new_slot in permuted_order]
        )
        restored[original_order] = float(score)
    if len(restored) != len(scores):
        raise ValueError("Candidate restoration produced duplicate orders")
    return restored


def aggregate_candidate_scores(
    views: Sequence[Mapping[Sequence[int], float]],
) -> dict[CandidateOrder, float]:
    if not views:
        raise ValueError("at least one TTA score map is required")
    normalized = [
        {as_candidate_order(order): float(score) for order, score in view.items()}
        for view in views
    ]
    expected = set(normalized[0])
    if any(set(view) != expected for view in normalized[1:]):
        raise ValueError("all TTA views must score the same candidates")
    return {
        order: sum(view[order] for view in normalized) / len(normalized)
        for order in expected
    }


def rank_candidate_scores(
    scores: Mapping[Sequence[int], float],
) -> list[tuple[CandidateOrder, float]]:
    values = [
        (as_candidate_order(order), float(score)) for order, score in scores.items()
    ]
    return sorted(values, key=lambda item: (-item[1], item[0]))
