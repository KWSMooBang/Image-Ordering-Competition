"""Pairwise multimodal ordering model.

The model compares two images in the context of a sentence and returns one
logit. A positive logit means image A is predicted to occur before image B.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PairwiseOrderingModel(nn.Module):
    """SigLIP-compatible pairwise ordering classifier."""

    def __init__(
        self,
        backbone_name: str,
        *,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install it with: pip install transformers"
            ) from exc

        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)

        image_dim = self._infer_projection_dim("vision")
        text_dim = self._infer_projection_dim("text")

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # [A, B, A-B, A*B, text] = 5 * projection_dim
        fusion_dim = projection_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _infer_projection_dim(self, modality: str) -> int:
        config = self.backbone.config

        candidates: list[Any]
        if modality == "vision":
            candidates = [
                getattr(config, "projection_dim", None),
                getattr(getattr(config, "vision_config", None), "projection_dim", None),
                getattr(getattr(config, "vision_config", None), "hidden_size", None),
                getattr(config, "hidden_size", None),
            ]
        else:
            candidates = [
                getattr(config, "projection_dim", None),
                getattr(getattr(config, "text_config", None), "projection_dim", None),
                getattr(getattr(config, "text_config", None), "hidden_size", None),
                getattr(config, "hidden_size", None),
            ]

        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value

        raise ValueError(
            f"Could not infer {modality} feature dimension from "
            f"{type(config).__name__}. Set a supported vision-language backbone."
        )

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @staticmethod
    def _extract_pooled(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        for name in ("pooler_output", "image_embeds", "text_embeds"):
            value = getattr(output, name, None)
            if isinstance(value, torch.Tensor):
                return value

        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden.mean(dim=1)

        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first.mean(dim=1) if first.ndim == 3 else first

        raise TypeError(f"Could not extract pooled features from {type(output).__name__}.")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "get_image_features"):
            features = self.backbone.get_image_features(pixel_values=pixel_values)
        elif hasattr(self.backbone, "vision_model"):
            features = self.backbone.vision_model(pixel_values=pixel_values)
        else:
            features = self.backbone(pixel_values=pixel_values)

        features = self._extract_pooled(features)
        return self.image_projection(features)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "get_text_features"):
            features = self.backbone.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif hasattr(self.backbone, "text_model"):
            features = self.backbone.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            features = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        features = self._extract_pooled(features)
        return self.text_projection(features)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_a = self.encode_image(image_a)
        feature_b = self.encode_image(image_b)
        text_feature = self.encode_text(input_ids, attention_mask)

        fused = torch.cat(
            [
                feature_a,
                feature_b,
                feature_a - feature_b,
                feature_a * feature_b,
                text_feature,
            ],
            dim=-1,
        )
        return self.classifier(fused).squeeze(-1)
