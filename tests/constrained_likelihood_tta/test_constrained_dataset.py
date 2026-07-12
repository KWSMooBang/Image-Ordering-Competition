"""Tests for constrained likelihood training records."""

import json

import pandas as pd

from src.constrained_likelihood_tta.dataset import build_training_records


def test_training_records_keep_caption_alignment_after_shuffle(tmp_path):
    row = {
        "Id": "sample-1",
        "Input_1": "a.jpg",
        "Input_2": "b.jpg",
        "Input_3": "c.jpg",
        "Input_4": "d.jpg",
        "Sentence": "A person completes an action.",
        "Answer": "[3, 1, 4, 2]",
        "No_ordering": False,
    }
    pd.DataFrame([row]).to_csv(tmp_path / "train.csv", index=False)
    cache_path = tmp_path / "captions.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        for image_index, image in enumerate(
            ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], start=1
        ):
            handle.write(
                json.dumps(
                    {
                        "Id": "sample-1",
                        "image_index": image_index,
                        "image": image,
                        "caption": f"caption {image}",
                    }
                )
                + "\n"
            )

    records = build_training_records(
        data_dir=tmp_path,
        train_csv_path=None,
        caption_cache_path=cache_path,
        caption_missing_policy="fail",
        max_samples=None,
        shuffle_augmentations_per_sample=1,
        shuffle_seed=3,
        shuffle_keep_original=True,
    )

    assert len(records) == 2
    for record in records:
        images = [record.row[f"Input_{index}"] for index in range(1, 5)]
        assert record.captions == [f"caption {image}" for image in images]
