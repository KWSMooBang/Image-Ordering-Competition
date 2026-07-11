import json

import pandas as pd

from src.pairwise_compare.make_pairs import (
    answer_to_order,
    build_pairwise_dataframe,
    build_test_pairwise_dataframe,
    order_to_answer,
)


def make_row(**overrides):
    row = {
        "Id": "sample_001",
        "Sentence": "A person performs an action.",
        "Answer": "[3, 2, 4, 1]",
        "No_ordering": False,
        "Input_1": "1.jpg",
        "Input_2": "2.jpg",
        "Input_3": "3.jpg",
        "Input_4": "4.jpg",
    }
    row.update(overrides)
    return row


def write_caption_cache(path):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, 5):
            handle.write(
                json.dumps(
                    {
                        "Id": "sample_001",
                        "image_index": index,
                        "image": f"{index}.jpg",
                        "caption": f"caption {index}",
                    }
                )
                + "\n"
            )


def test_answer_order_round_trip():
    answer = [3, 2, 4, 1]
    order = answer_to_order(answer)

    assert order == [4, 2, 1, 3]
    assert order_to_answer(order) == answer


def test_build_pairs_includes_image_captions(tmp_path):
    cache_path = tmp_path / "captions.jsonl"
    write_caption_cache(cache_path)

    pairs = build_pairwise_dataframe(
        pd.DataFrame([make_row()]),
        pair_mode="canonical",
        caption_cache_path=cache_path,
        caption_missing_policy="fail",
    )

    assert len(pairs) == 6
    first = pairs.iloc[0]
    assert first["image_a_caption"] == "caption 1"
    assert first["image_b_caption"] == "caption 2"


def test_build_test_pairs_includes_image_captions(tmp_path):
    cache_path = tmp_path / "captions.jsonl"
    write_caption_cache(cache_path)
    test_row = make_row()
    test_row.pop("Answer")
    test_row.pop("No_ordering")

    pairs = build_test_pairwise_dataframe(
        pd.DataFrame([test_row]),
        caption_cache_path=cache_path,
        caption_missing_policy="fail",
    )

    assert len(pairs) == 6
    assert set(pairs["image_a_caption"]) >= {"caption 1", "caption 2", "caption 3"}
