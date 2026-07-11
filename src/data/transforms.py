"""Image preprocessing utilities.

For pretrained Hugging Face vision encoders such as SigLIP, prefer
``build_hf_image_transform`` so the model's own image processor performs the
required resize, rescale, and normalization.

The torchvision transforms are useful for a custom CNN/MLP image encoder.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EnsureRGB:
    """Convert PIL images to three-channel RGB."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}.")
        return image.convert("RGB")


class HuggingFaceImageTransform:
    """Wrap a Hugging Face image processor as a single-image transform."""

    def __init__(
        self,
        image_processor: Any,
        *,
        output_key: str = "pixel_values",
    ) -> None:
        self.image_processor = image_processor
        self.output_key = output_key

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        encoded = self.image_processor(images=image, return_tensors="pt")

        if self.output_key not in encoded:
            raise KeyError(
                f"Processor output has no {self.output_key!r}. "
                f"Available keys: {list(encoded.keys())}"
            )

        tensor = encoded[self.output_key]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        if tensor.ndim < 1 or tensor.shape[0] != 1:
            raise ValueError(
                f"Expected a single-item processor batch; got shape {tuple(tensor.shape)}."
            )
        return tensor.squeeze(0)


def _validate_normalisation(
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three channel values.")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be positive.")


def build_train_transform(
    image_size: int = 224,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    horizontal_flip_probability: float = 0.5,
    crop_scale: tuple[float, float] = (0.9, 1.0),
) -> Callable[[Image.Image], torch.Tensor]:
    """Build conservative training augmentation for custom vision encoders."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must be between 0 and 1.")
    _validate_normalisation(mean, std)

    from torchvision import transforms

    return transforms.Compose(
        [
            EnsureRGB(),
            transforms.RandomResizedCrop(
                size=image_size,
                scale=crop_scale,
                ratio=(0.95, 1.05),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build deterministic validation/test preprocessing."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    _validate_normalisation(mean, std)

    from torchvision import transforms

    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            EnsureRGB(),
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_hf_image_transform(
    model_name_or_path: str,
    *,
    output_key: str = "pixel_values",
    trust_remote_code: bool = False,
) -> HuggingFaceImageTransform:
    """Load a pretrained Hugging Face image processor and wrap it.

    This function imports Transformers lazily, so scripts that only use the
    CSV utilities do not require Transformers to be imported.
    """
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for build_hf_image_transform. "
            "Install it with: pip install transformers"
        ) from exc

    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return HuggingFaceImageTransform(
        image_processor=image_processor,
        output_key=output_key,
    )