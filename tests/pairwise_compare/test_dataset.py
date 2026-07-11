import pandas as pd
import torch
from PIL import Image

from src.pairwise_compare.dataset import PairwiseDataset


def test_dataset_composes_sentence_and_pair_captions(tmp_path):
    sample_dir = tmp_path / "sample_001"
    sample_dir.mkdir()
    for name in ("1.jpg", "2.jpg"):
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(sample_dir / name)

    pairs = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "pair_id": "sample_001__1_2",
                "sentence": "A person opens a box.",
                "image_a_index": 1,
                "image_b_index": 2,
                "image_a_path": "sample_001/1.jpg",
                "image_b_path": "sample_001/2.jpg",
                "image_a_caption": "closed box",
                "image_b_caption": "open box",
                "label": 1,
                "no_ordering": False,
            }
        ]
    )

    dataset = PairwiseDataset(
        pairs,
        image_root=tmp_path,
        transform=lambda image: torch.zeros(3, 4, 4),
    )
    sample = dataset[0]

    assert "Story sentence" in sample["text"]
    assert "Image A caption: closed box" in sample["text"]
    assert "Image B caption: open box" in sample["text"]
    assert sample["label"].item() == 1.0
