"""Tests for the independent constrained likelihood model helpers."""

from types import SimpleNamespace

import torch

from src.constrained_likelihood_tta.model import (
    ConstrainedQwenOrderer,
    build_candidate_token_sequences,
    qwen_model_loader_name,
    resolve_eos_token_ids,
    resolve_pad_token_id,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens, return_attention_mask):
        digits = [int(character) for character in text if character.isdigit()]
        return {"input_ids": [90, *digits, 91]}


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return FakeBatch({"input_ids": torch.tensor([[7, 8]])})


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    config = SimpleNamespace(eos_token_id=99, pad_token_id=0)

    def generate(self, **kwargs):
        prompt = kwargs["input_ids"]
        constraint = kwargs["prefix_allowed_tokens_fn"]
        sequences = [
            torch.tensor([*prompt[0].tolist(), *tokens])
            for tokens in constraint.sequences
        ]
        return SimpleNamespace(
            sequences=torch.stack(sequences),
            sequences_scores=torch.tensor(
                [-float(index + 1) for index in range(len(sequences))]
            ),
        )


def test_candidate_token_sequences_are_unique_and_end_with_eos():
    sequences = build_candidate_token_sequences(
        FakeTokenizer(),
        [(1, 2, 3, 4), (2, 1, 3, 4)],
        eos_token_id=99,
    )

    assert sequences[(1, 2, 3, 4)][-1] == 99
    assert sequences[(1, 2, 3, 4)] != sequences[(2, 1, 3, 4)]


def test_qwen_loader_and_special_token_resolution():
    assert qwen_model_loader_name("Qwen/Qwen3.5-4B") == "AutoModelForMultimodalLM"
    assert (
        qwen_model_loader_name("Qwen/Qwen3-VL-8B-Instruct")
        == "Qwen3VLForConditionalGeneration"
    )

    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[98, 99], pad_token_id=None),
        config=SimpleNamespace(eos_token_id=97, pad_token_id=None),
    )
    tokenizer = FakeTokenizer()
    assert resolve_eos_token_ids(model, tokenizer) == [98, 99]
    assert resolve_pad_token_id(model, tokenizer, [98, 99]) == 0


def test_orderer_scores_every_candidate_across_small_beam_groups():
    orderer = ConstrainedQwenOrderer.__new__(ConstrainedQwenOrderer)
    orderer.processor = FakeProcessor()
    orderer.model = FakeModel()
    orderer.torch = torch
    candidates = [
        (1, 2, 3, 4),
        (2, 1, 3, 4),
        (3, 1, 2, 4),
        (4, 1, 2, 3),
    ]

    results = orderer.score_candidates(
        [],
        candidates,
        candidate_batch_size=4,
        normalization="sum",
    )

    assert [result.order for result in results] == candidates
    assert all(result.token_count == 7 for result in results)
