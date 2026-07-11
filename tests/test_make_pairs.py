import pandas as pd

from src.data.make_pairs import (
    answer_to_order,
    build_pairwise_dataframe,
    order_to_answer,
)


def test_answer_order_round_trip():
    answer = [3, 2, 4, 1]
    order = answer_to_order(answer)

    assert order == [4, 2, 1, 3]
    assert order_to_answer(order) == answer


def test_build_six_canonical_pairs():
    source = pd.DataFrame(
        [
            {
                "Id": "sample_001",
                "Sentence": "A person performs an action.",
                "Answer": "[3, 2, 4, 1]",
                "No_ordering": False,
                "Input_1": "1.jpg",
                "Input_2": "2.jpg",
                "Input_3": "3.jpg",
                "Input_4": "4.jpg",
            }
        ]
    )

    pairs = build_pairwise_dataframe(source, pair_mode="canonical")

    assert len(pairs) == 6
    assert pairs["sample_id"].nunique() == 1

    labels = {
        (row.image_a_index, row.image_b_index): row.label
        for row in pairs.itertuples(index=False)
    }

    # Actual chronological order is 4 -> 2 -> 1 -> 3.
    assert labels[(1, 2)] == 0
    assert labels[(1, 3)] == 1
    assert labels[(1, 4)] == 0
    assert labels[(2, 3)] == 1
    assert labels[(2, 4)] == 0
    assert labels[(3, 4)] == 0
