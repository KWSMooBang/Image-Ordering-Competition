import pandas as pd
import pytest

from src.caption_augmented.tta import (
    build_tta_permutations,
    consensus_chronological_order,
    permute_row_and_captions,
    restore_chronological_order,
)


def make_row() -> pd.Series:
    return pd.Series(
        {
            "Id": "sample-1",
            "Input_1": "a.jpg",
            "Input_2": "b.jpg",
            "Input_3": "c.jpg",
            "Input_4": "d.jpg",
            "Sentence": "A short story.",
        }
    )


def test_build_tta_permutations_is_deterministic_unique_and_identity_first():
    first = build_tta_permutations(8, seed=7)
    second = build_tta_permutations(8, seed=7)

    assert first == second
    assert first[0] == [1, 2, 3, 4]
    assert len({tuple(order) for order in first}) == 8


def test_default_four_tta_views_cover_every_original_image_in_every_slot():
    views = build_tta_permutations(4)

    for slot in range(4):
        assert sorted(view[slot] for view in views) == [1, 2, 3, 4]


@pytest.mark.parametrize("count", [0, 25])
def test_build_tta_permutations_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="between 1 and 24"):
        build_tta_permutations(count)


def test_permute_row_and_captions_keeps_image_caption_alignment():
    row, captions = permute_row_and_captions(
        make_row(),
        ["caption a", "caption b", "caption c", "caption d"],
        [2, 4, 1, 3],
    )

    assert [row[f"Input_{index}"] for index in range(1, 5)] == ["b.jpg", "d.jpg", "a.jpg", "c.jpg"]
    assert captions == ["caption b", "caption d", "caption a", "caption c"]


def test_restore_chronological_order_maps_permuted_labels_to_original_labels():
    assert restore_chronological_order([3, 1, 4, 2], [2, 4, 1, 3]) == [1, 2, 3, 4]


def test_consensus_uses_majority_vote_and_reports_winning_vote_count():
    order, votes = consensus_chronological_order(
        [[2, 1, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4], [4, 3, 2, 1]]
    )

    assert order == [2, 1, 3, 4]
    assert votes == 2


def test_consensus_breaks_tie_using_earliest_view():
    order, votes = consensus_chronological_order([[3, 1, 2, 4], [1, 2, 3, 4]])

    assert order == [3, 1, 2, 4]
    assert votes == 1


def test_consensus_requires_a_valid_prediction():
    with pytest.raises(ValueError, match="At least one valid"):
        consensus_chronological_order([])
