from pathlib import Path

import pytest
import torch

from src.caption_augmented.dataset import OrderTrainingRecord
from src.caption_augmented.train import OrdererSFTCollator, is_main_process, parse_lora_target_modules, resolve_device_map


class FakeBatch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FakeProcessor:
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors):
        length = 6 if add_generation_prompt else 10
        ids = torch.arange(1, length + 1).unsqueeze(0)
        return FakeBatch({"input_ids": ids})


def make_record() -> OrderTrainingRecord:
    return OrderTrainingRecord(
        row={
            "Id": "sample-1",
            "Input_1": "a.jpg",
            "Input_2": "b.jpg",
            "Input_3": "c.jpg",
            "Input_4": "d.jpg",
            "Sentence": "A person opens a box and takes out a cup.",
        },
        captions=["caption 1", "caption 2", "caption 3", "caption 4"],
        target_text="[4, 2, 1, 3]",
    )


def test_parse_lora_target_modules_trims_empty_values():
    assert parse_lora_target_modules("q_proj, k_proj,,v_proj ") == ["q_proj", "k_proj", "v_proj"]


def test_resolve_device_map_supports_torchrun_local_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert resolve_device_map("local") == {"": 2}
    assert resolve_device_map("none") is None
    assert resolve_device_map("auto") == "auto"


def test_is_main_process_uses_rank_env(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    assert not is_main_process()

    monkeypatch.setenv("RANK", "0")
    assert is_main_process()


def test_orderer_sft_collator_masks_prompt_tokens():
    collator = OrdererSFTCollator(processor=FakeProcessor(), image_dir=Path("/data/train"))
    batch = collator([make_record()])

    assert batch["input_ids"].shape == (1, 10)
    assert batch["labels"][0, :6].tolist() == [-100] * 6
    assert batch["labels"][0, 6:].tolist() == [7, 8, 9, 10]


def test_orderer_sft_collator_requires_batch_size_one():
    collator = OrdererSFTCollator(processor=FakeProcessor(), image_dir=Path("/data/train"))
    with pytest.raises(ValueError, match="batch size 1"):
        collator([make_record(), make_record()])
