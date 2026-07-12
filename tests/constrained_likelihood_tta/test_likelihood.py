import pytest

from src.constrained_likelihood_tta.likelihood import (
    ALL_CANDIDATE_ORDERS,
    CandidateTokenConstraint,
    aggregate_candidate_scores,
    chunk_candidates,
    rank_candidate_scores,
    restore_candidate_scores,
)


def test_candidate_space_contains_all_24_permutations_including_identity():
    assert len(ALL_CANDIDATE_ORDERS) == 24
    assert len(set(ALL_CANDIDATE_ORDERS)) == 24
    assert (1, 2, 3, 4) in ALL_CANDIDATE_ORDERS


def test_candidate_chunks_never_leave_a_single_beam_group():
    chunks = chunk_candidates(ALL_CANDIDATE_ORDERS, 4)

    assert [len(chunk) for chunk in chunks] == [4] * 6
    assert all(len({order[0] for order in chunk}) == len(chunk) for chunk in chunks)
    assert {order for chunk in chunks for order in chunk} == set(ALL_CANDIDATE_ORDERS)


def test_token_constraint_follows_only_candidate_prefixes():
    constraint = CandidateTokenConstraint(
        [[10, 11, 99], [10, 12, 99]],
        prompt_length=2,
    )

    assert constraint.allowed_next_tokens([1, 2]) == [10]
    assert constraint.allowed_next_tokens([1, 2, 10]) == [11, 12]
    assert constraint.allowed_next_tokens([1, 2, 10, 11]) == [99]
    with pytest.raises(RuntimeError, match="outside"):
        constraint.allowed_next_tokens([1, 2, 77])


def test_restore_and_average_candidate_scores_across_tta_views():
    first = {
        (3, 1, 4, 2): -1.0,
        (1, 2, 3, 4): -3.0,
    }
    restored = restore_candidate_scores(first, [2, 4, 1, 3])

    assert restored[(1, 2, 3, 4)] == -1.0
    averaged = aggregate_candidate_scores(
        [
            restored,
            {(1, 2, 3, 4): -2.0, (2, 4, 1, 3): -4.0},
        ]
    )
    assert averaged[(1, 2, 3, 4)] == -1.5
    assert rank_candidate_scores(averaged)[0][0] == (1, 2, 3, 4)
