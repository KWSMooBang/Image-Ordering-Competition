from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.constrained_likelihood_tta.likelihood import (
    CandidateLikelihood,
    CandidateOrder,
    CandidateTokenConstraint,
    as_candidate_order,
    candidate_text,
    chunk_candidates,
)

QWEN35_LOADER = "AutoModelForMultimodalLM"
QWEN3_VL_LOADER = "Qwen3VLForConditionalGeneration"


class Captioner(Protocol):
    def caption(
        self, image_path: Path, prompt: str | None, max_new_tokens: int
    ) -> str: ...


def qwen_model_loader_name(model_name: str) -> str:
    normalized = model_name.lower()
    if "qwen3-vl" in normalized:
        return QWEN3_VL_LOADER
    return QWEN35_LOADER


def _torch_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def load_qwen_bundle(
    model_name: str,
    *,
    device_map: str | dict[str, int] | None,
    torch_dtype: str,
    attn_implementation: str | None,
    quantization_config: Any = None,
):
    import torch
    import transformers
    from transformers import AutoProcessor

    loader_name = qwen_model_loader_name(model_name)
    try:
        model_cls = getattr(transformers, loader_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"{model_name} requires transformers with {loader_name} support"
        ) from exc

    kwargs: dict[str, Any] = {"dtype": _torch_dtype(torch, torch_dtype)}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    processor = AutoProcessor.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, **kwargs)
    return processor, model


def apply_adapter(model: Any, adapter_path: str | Path | None) -> Any:
    if adapter_path is None:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("PEFT is required to load a trained adapter") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


class BlipCaptioner:
    def __init__(self, model_name: str, *, device: str, torch_dtype: str) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA captioning was requested but CUDA is unavailable")
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if torch_dtype != "auto":
            dtype = _torch_dtype(torch, torch_dtype)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.torch = torch

    def caption(self, image_path: Path, prompt: str | None, max_new_tokens: int) -> str:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        inputs = (
            self.processor(rgb, prompt, return_tensors="pt")
            if prompt
            else self.processor(
                rgb,
                return_tensors="pt",
            )
        )
        inputs = inputs.to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)


class QwenCaptioner:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str,
        torch_dtype: str,
        attn_implementation: str | None,
    ) -> None:
        self.processor, self.model = load_qwen_bundle(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model.eval()

    def caption(self, image_path: Path, prompt: str | None, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {
                        "type": "text",
                        "text": prompt or "Describe this image concisely.",
                    },
                ],
            }
        ]
        return _generate_text(self.processor, self.model, messages, max_new_tokens)


class ConstrainedQwenOrderer:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str,
        torch_dtype: str,
        attn_implementation: str | None,
        adapter_path: str | Path | None,
    ) -> None:
        import torch

        self.processor, self.model = load_qwen_bundle(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model = apply_adapter(self.model, adapter_path)
        self.model.eval()
        self.torch = torch

    def score_candidates(
        self,
        messages: list[dict[str, object]],
        candidates: Sequence[Sequence[int]],
        *,
        candidate_batch_size: int,
        normalization: str,
    ) -> list[CandidateLikelihood]:
        if normalization not in {"sum", "mean"}:
            raise ValueError("normalization must be 'sum' or 'mean'")
        normalized = [as_candidate_order(candidate) for candidate in candidates]
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate orders must be unique")

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)
        prompt_length = int(inputs["input_ids"].shape[1])

        eos_token_ids = resolve_eos_token_ids(self.model, self.processor.tokenizer)
        pad_token_id = resolve_pad_token_id(
            self.model, self.processor.tokenizer, eos_token_ids
        )
        token_sequences = build_candidate_token_sequences(
            self.processor.tokenizer,
            normalized,
            eos_token_id=eos_token_ids[0],
        )

        results: dict[CandidateOrder, CandidateLikelihood] = {}
        for group in chunk_candidates(normalized, candidate_batch_size):
            group_tokens = [token_sequences[order] for order in group]
            constraint = CandidateTokenConstraint(
                group_tokens, prompt_length=prompt_length
            )
            with self.torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max(len(tokens) for tokens in group_tokens),
                    num_beams=len(group),
                    num_return_sequences=len(group),
                    do_sample=False,
                    early_stopping=True,
                    length_penalty=0.0,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=constraint,
                    eos_token_id=eos_token_ids,
                    pad_token_id=pad_token_id,
                    renormalize_logits=False,
                )

            sequence_scores = getattr(output, "sequences_scores", None)
            if sequence_scores is None:
                raise RuntimeError("Beam decoding did not return sequence likelihoods")
            target_to_order = {tuple(token_sequences[order]): order for order in group}
            for sequence, raw_score in zip(
                output.sequences, sequence_scores, strict=True
            ):
                generated = tuple(
                    int(token) for token in sequence[prompt_length:].tolist()
                )
                matches = [
                    (tokens, order)
                    for tokens, order in target_to_order.items()
                    if generated[: len(tokens)] == tokens
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Could not map constrained output to one candidate: {generated}"
                    )
                tokens, order = matches[0]
                log_likelihood = float(raw_score.item())
                score = (
                    log_likelihood
                    if normalization == "sum"
                    else log_likelihood / len(tokens)
                )
                results[order] = CandidateLikelihood(
                    order=order,
                    log_likelihood=log_likelihood,
                    token_count=len(tokens),
                    score=score,
                )

        if set(results) != set(normalized):
            missing = sorted(set(normalized) - set(results))
            raise RuntimeError(
                f"Likelihood decoding missed candidate orders: {missing}"
            )
        return [results[order] for order in normalized]


def _token_ids(encoded: Any) -> list[int]:
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token) for token in values]


def build_candidate_token_sequences(
    tokenizer: Any,
    candidates: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
) -> dict[CandidateOrder, list[int]]:
    sequences: dict[CandidateOrder, list[int]] = {}
    for candidate in candidates:
        order = as_candidate_order(candidate)
        encoded = tokenizer(
            candidate_text(order),
            add_special_tokens=False,
            return_attention_mask=False,
        )
        tokens = _token_ids(encoded)
        if not tokens:
            raise RuntimeError(f"Tokenizer produced no tokens for candidate {order}")
        if tokens[-1] != eos_token_id:
            tokens.append(int(eos_token_id))
        sequences[order] = tokens
    if len({tuple(tokens) for tokens in sequences.values()}) != len(sequences):
        raise RuntimeError("Two candidates produced the same token sequence")
    return sequences


def _normalize_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    return [int(token) for token in value]


def resolve_eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    for value in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        token_ids = _normalize_token_ids(value)
        if token_ids:
            return list(dict.fromkeys(token_ids))
    raise RuntimeError("Could not resolve EOS token ids")


def resolve_pad_token_id(
    model: Any, tokenizer: Any, eos_token_ids: Sequence[int]
) -> int:
    for value in (
        getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        token_ids = _normalize_token_ids(value)
        if token_ids:
            return token_ids[0]
    return int(eos_token_ids[0])


def _generate_text(
    processor: Any,
    model: Any,
    messages: list[dict[str, object]],
    max_new_tokens: int,
) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = inputs.to(model_device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
