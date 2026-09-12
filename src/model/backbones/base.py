"""Shared backbone construction types and path handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BackboneConfig:
    kind: str
    model_name: str
    pretrained: bool
    pretrained_weights: str | Path | None
    image_size: int
    drop_path_rate: float
    trainable_last_n_blocks: int
    lora_last_n_blocks: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float


@dataclass(frozen=True)
class BuiltBackbone:
    module: nn.Module
    feature_dim: int


def resolve_pretrained_weights(path: str | Path) -> Path:
    """Resolve a local checkpoint independently of the working directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"pretrained weights do not exist: {candidate}")
    return candidate
