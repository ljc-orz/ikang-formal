"""ConvNeXt image and demographic metadata binary classifier."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED_WEIGHTS = Path(
    "pretrained-weights/convnext_tiny.fb_in22k_ft_in1k/pytorch_model.bin"
)


def resolve_pretrained_weights(path: str | Path) -> Path:
    """Resolve a local pretrained checkpoint independently of the working directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"pretrained weights do not exist: {candidate}")
    return candidate


class FundusClassifier(nn.Module):
    """Fuse a fundus embedding with normalized age and binary sex metadata."""

    def __init__(
        self,
        model_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
        pretrained_weights: str | Path | None = DEFAULT_PRETRAINED_WEIGHTS,
        metadata_hidden_dim: int = 16,
        classifier_dropout: float = 0.2,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        pretrained_cfg_overlay = None
        if pretrained:
            if pretrained_weights is None:
                raise ValueError(
                    "pretrained=True requires a local pretrained_weights path"
                )
            weights_path = resolve_pretrained_weights(pretrained_weights)
            pretrained_cfg_overlay = {"file": str(weights_path)}
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        if not hasattr(self.backbone, "stages") or len(self.backbone.stages) < 2:
            raise ValueError(
                f"V1 requires a ConvNeXt-like backbone with at least two stages: {model_name}"
            )

        image_dim = int(self.backbone.num_features)
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
        self.freeze_for_v1()

    def freeze_for_v1(self) -> None:
        """Train only the final two backbone stages, pooling head and new heads."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for stage in self.backbone.stages[-2:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
        if hasattr(self.backbone, "head"):
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True
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
