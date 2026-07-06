from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image


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
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_kwargs = {"device_map": device_map}
        if torch_dtype == "auto":
            model_kwargs["dtype"] = "auto"
        elif torch_dtype == "float16":
            model_kwargs["dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        elif torch_dtype == "float32":
            model_kwargs["dtype"] = torch.float32
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
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
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_kwargs = {"device_map": device_map}
        if torch_dtype == "auto":
            model_kwargs["dtype"] = "auto"
        elif torch_dtype == "float16":
            model_kwargs["dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            model_kwargs["dtype"] = torch.bfloat16
        elif torch_dtype == "float32":
            model_kwargs["dtype"] = torch.float32
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

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
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
