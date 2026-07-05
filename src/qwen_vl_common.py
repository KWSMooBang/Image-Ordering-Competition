from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class LoadedVLModel:
    torch: Any
    processor: Any
    model: Any
    family: str
    process_vision_info: Any | None = None


def detect_qwen_family(model_name: str) -> str:
    normalized = model_name.lower().replace("_", "-")
    if "qwen3-vl" in normalized:
        return "qwen3-vl"
    if "qwen2.5-vl" in normalized or "qwen2-5-vl" in normalized:
        return "qwen2.5-vl"
    if "qwen2-vl" in normalized:
        return "qwen2-vl"
    raise ValueError(
        "Unsupported model family. Use a Qwen VL checkpoint such as "
        "Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, "
        "or Qwen/Qwen3-VL-8B-Instruct."
    )


def get_processor_kwargs(min_pixels: int | None, max_pixels: int | None) -> dict[str, int]:
    processor_kwargs: dict[str, int] = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    return processor_kwargs


def load_peft_adapter(model: Any, adapter_path: str | Path | None) -> Any:
    if adapter_path is None:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Loading a LoRA adapter requires `peft`.") from exc
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def load_qwen_vl(
    model_name: str,
    *,
    attn_implementation: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    load_in_4bit: bool = False,
    adapter_path: str | Path | None = None,
) -> LoadedVLModel:
    family = detect_qwen_family(model_name)
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Qwen VLM dependencies are missing. Run `bash init.sh`.") from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit inference requires `bitsandbytes`.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor_kwargs = get_processor_kwargs(min_pixels=min_pixels, max_pixels=max_pixels)

    if family == "qwen3-vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires a recent transformers build. "
                "Install the latest transformers package, or use "
                "Qwen/Qwen2.5-VL-7B-Instruct as a fallback."
            ) from exc
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        model = load_peft_adapter(model, adapter_path)
        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        return LoadedVLModel(torch=torch, processor=processor, model=model, family=family)

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("Missing `qwen-vl-utils`. Run `bash init.sh`.") from exc

    if family == "qwen2.5-vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires a recent transformers build. "
                "Install the latest transformers package or use Qwen/Qwen2-VL-2B-Instruct."
            ) from exc
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    else:
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen2-VL support is unavailable in this transformers install.") from exc
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    model = load_peft_adapter(model, adapter_path)
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    return LoadedVLModel(
        torch=torch,
        processor=processor,
        model=model,
        family=family,
        process_vision_info=process_vision_info,
    )


def generate_text(bundle: LoadedVLModel, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    if bundle.family == "qwen3-vl":
        inputs = bundle.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        if bundle.process_vision_info is None:
            raise RuntimeError("Qwen2-style generation requires process_vision_info.")
        text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = bundle.process_vision_info(messages)
        inputs = bundle.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(bundle.model.device)
    with bundle.torch.no_grad():
        generated_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return bundle.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
