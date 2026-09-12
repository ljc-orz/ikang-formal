"""Image-backbone and demographic metadata binary classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .backbones import BackboneConfig, build_backbone, resolve_pretrained_weights

DEFAULT_PRETRAINED_WEIGHTS = Path(
    "pretrained-weights/convnext_tiny.fb_in22k_ft_in1k/pytorch_model.bin"
)
DEFAULT_RETFOUND_WEIGHTS = Path(
    "pretrained-weights/RETFound_dinov2_shanghai/"
    "RETFound_dinov2_shanghai_backbone.pth"
)


class FundusClassifier(nn.Module):
    """Fuse a fundus embedding with normalized age and binary sex metadata."""

    def __init__(
        self,
        backbone_type: str = "convnext",
        model_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
        pretrained_weights: str | Path | None = DEFAULT_PRETRAINED_WEIGHTS,
        retfound_pretrained_weights: str | Path | None = DEFAULT_RETFOUND_WEIGHTS,
        image_size: int = 224,
        metadata_hidden_dim: int = 16,
        classifier_dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        trainable_last_n_blocks: int = 3,
        lora_last_n_blocks: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        selected_weights = (
            retfound_pretrained_weights
            if backbone_type == "retfound_dinov2"
            else pretrained_weights
        )
        built = build_backbone(
            BackboneConfig(
                kind=backbone_type,
                model_name=model_name,
                pretrained=pretrained,
                pretrained_weights=selected_weights,
                image_size=image_size,
                drop_path_rate=drop_path_rate,
                trainable_last_n_blocks=trainable_last_n_blocks,
                lora_last_n_blocks=lora_last_n_blocks,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        )
        self.backbone = built.module
        self.backbone_type = backbone_type

        image_dim = built.feature_dim
        self.meta_encoder = nn.Sequential(
            nn.Linear(2, metadata_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_dim + metadata_hidden_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(image_dim + metadata_hidden_dim, 1),
        )
        for module in (self.meta_encoder, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(
        self, image: torch.Tensor, age: torch.Tensor, sex: torch.Tensor
    ) -> torch.Tensor:
        image_feature = self.backbone(image)
        # Age normalization keeps the two metadata dimensions on similar scales.
        metadata = torch.stack((age.float() / 100.0, sex.float()), dim=1)
        meta_feature = self.meta_encoder(metadata)
        return self.classifier(torch.cat((image_feature, meta_feature), dim=1)).squeeze(1)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return trainable, total
