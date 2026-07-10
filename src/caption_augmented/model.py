from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PIL import Image

QWEN35_LOADER = "AutoModelForMultimodalLM"
QWEN3_VL_LOADER = "Qwen3VLForConditionalGeneration"


class Captioner(Protocol):
    def caption(self, image_path: Path, prompt: str | None = None, max_new_tokens: int = 64) -> str:
        ...


class Orderer(Protocol):
    def generate_order(self, messages: list[dict[str, object]], max_new_tokens: int) -> str:
        ...


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def qwen_model_loader_name(model_name: str) -> str:
    normalized = model_name.lower()
    if "qwen3.5" in normalized:
        return QWEN35_LOADER
    if "qwen3-vl" in normalized:
        return QWEN3_VL_LOADER
    return QWEN35_LOADER


def resolve_qwen_dtype(torch: Any, torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]


def build_qwen_model_kwargs(
    torch: Any,
    *,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": resolve_qwen_dtype(torch, torch_dtype),
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    return model_kwargs


def load_qwen_processor_and_model(model_name: str, model_kwargs: dict[str, Any]):
    try:
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen loading requires torch and a recent transformers build. "
            "Qwen3.5 models specifically require `AutoModelForMultimodalLM` support."
        ) from exc

    loader_name = qwen_model_loader_name(model_name)
    try:
        model_cls = getattr(transformers, loader_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"{model_name} requires transformers with `{loader_name}`. "
            "Install or upgrade to a recent transformers build on the GPU machine."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    return processor, model


def apply_peft_adapter(model: Any, adapter_path: str | Path | None) -> Any:
    if adapter_path is None:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Loading a LoRA adapter for inference requires `peft`.") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


class BlipCaptioner:
    def __init__(self, model_name: str, device: str = "auto", torch_dtype: str = "auto"):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if torch_dtype == "float32":
            dtype = torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.torch = torch

    def caption(self, image_path: Path, prompt: str | None = None, max_new_tokens: int = 64) -> str:
        with Image.open(image_path) as image:
            raw_image = image.convert("RGB")

        if prompt:
            inputs = self.processor(raw_image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(raw_image, return_tensors="pt")
        inputs = inputs.to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)


class QwenCaptioner:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
    ):
        import torch

        model_kwargs = build_qwen_model_kwargs(
            torch,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.processor, self.model = load_qwen_processor_and_model(model_name, model_kwargs)
        self.torch = torch

    def caption(self, image_path: Path, prompt: str | None = None, max_new_tokens: int = 64) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt or "Describe this image in one concise sentence."},
                ],
            }
        ]
        return _generate_qwen_text(self.processor, self.model, messages, max_new_tokens=max_new_tokens)


class QwenOrderer:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
        adapter_path: str | Path | None = None,
    ):
        import torch

        model_kwargs = build_qwen_model_kwargs(
            torch,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.processor, self.model = load_qwen_processor_and_model(model_name, model_kwargs)
        self.model = apply_peft_adapter(self.model, adapter_path)
        self.model.eval()

    def generate_order(self, messages: list[dict[str, object]], max_new_tokens: int) -> str:
        return _generate_qwen_text(self.processor, self.model, messages, max_new_tokens=max_new_tokens)


def _generate_qwen_text(processor, model, messages: list[dict[str, object]], max_new_tokens: int) -> str:
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
