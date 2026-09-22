"""Reconstruct the exact project model stored in a training checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.model import FundusClassifier


SUPPORTED_BACKBONES = ("resnet50", "retfound_dinov2")


@dataclass(frozen=True)
class LoadedFundusModel:
    model: FundusClassifier
    backbone_type: str
    target: str
    threshold: float
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    checkpoint_path: Path
    config: Mapping[str, Any]


def _build_model(config: Mapping[str, Any]) -> FundusClassifier:
    data = config["data"]
    model_config = config["model"]
    backbone_type = str(model_config.get("backbone", "convnext"))
    if backbone_type not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"heatmap visualization supports {SUPPORTED_BACKBONES}, "
            f"but checkpoint uses {backbone_type!r}"
        )
    # The complete fine-tuned state is loaded below. Disabling pretrained here
    # avoids loading a second checkpoint and still creates the exact LoRA/base
    # module hierarchy used during training.
    return FundusClassifier(
        backbone_type=backbone_type,
        model_name=str(model_config["name"]),
        pretrained=False,
        image_size=int(data["image_size"]),
        metadata_hidden_dim=int(model_config["metadata_hidden_dim"]),
        classifier_dropout=float(model_config["classifier_dropout"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
        trainable_last_n_blocks=int(
            model_config.get("trainable_last_n_blocks", 3)
        ),
        lora_last_n_blocks=int(model_config.get("lora_last_n_blocks", 2)),
        lora_rank=int(model_config.get("lora_rank", 8)),
        lora_alpha=float(model_config.get("lora_alpha", 16.0)),
        lora_dropout=float(model_config.get("lora_dropout", 0.0)),
    )


def load_fundus_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str
) -> LoadedFundusModel:
    """Strictly load a training checkpoint and return heatmap metadata."""
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    for key in ("model_state", "config", "target"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing {key!r}: {path}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint config must be a mapping: {path}")

    model = _build_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)

    data = config["data"]
    backbone_type = str(config["model"].get("backbone", "convnext"))
    threshold = float(checkpoint.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"checkpoint threshold must be in [0, 1], got {threshold}")
    return LoadedFundusModel(
        model=model,
        backbone_type=backbone_type,
        target=str(checkpoint["target"]),
        threshold=threshold,
        image_size=int(data["image_size"]),
        mean=tuple(float(value) for value in data["mean"]),
        std=tuple(float(value) for value in data["std"]),
        checkpoint_path=path,
        config=config,
    )
