"""RETFound-DINOv2 ViT-L/14 backbone with LoRA on its final blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .base import BackboneConfig, BuiltBackbone, resolve_pretrained_weights
from .lora import LoRALinear


ARCHITECTURE = "vit_large_patch14_dinov2.lvd142m"
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _create_model(config: BackboneConfig, *, meta: bool) -> nn.Module:
    arguments = {
        "pretrained": False,
        "img_size": config.image_size,
        "num_classes": 0,
        "drop_path_rate": config.drop_path_rate,
    }
    if meta:
        with torch.device("meta"):
            return timm.create_model(ARCHITECTURE, **arguments)
    return timm.create_model(ARCHITECTURE, **arguments)


def _interpolate_pos_embed(
    value: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor:
    """Match the bicubic interpolation used by the official RETFound code."""
    if value.shape == target_shape:
        return value
    extra_tokens = 1
    source_grid = math.isqrt(value.shape[1] - extra_tokens)
    target_grid = math.isqrt(target_shape[1] - extra_tokens)
    if (
        value.ndim != 3
        or source_grid * source_grid != value.shape[1] - extra_tokens
        or target_grid * target_grid != target_shape[1] - extra_tokens
        or value.shape[2] != target_shape[2]
    ):
        raise ValueError(
            f"cannot interpolate pos_embed from {tuple(value.shape)} "
            f"to {tuple(target_shape)}"
        )
    prefix = value[:, :extra_tokens]
    patches = value[:, extra_tokens:].reshape(
        1, source_grid, source_grid, value.shape[2]
    )
    patches = F.interpolate(
        patches.permute(0, 3, 1, 2),
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat(
        (prefix, patches.permute(0, 2, 3, 1).flatten(1, 2)), dim=1
    )


def _load_backbone_state(
    config: BackboneConfig, expected: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if config.pretrained_weights is None:
        raise ValueError(
            "pretrained RETFound requires model.retfound_pretrained_weights"
        )
    path = resolve_pretrained_weights(config.pretrained_weights)
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"RETFound checkpoint is not a state dictionary: {path}")

    teacher = checkpoint.get("teacher")
    if isinstance(teacher, Mapping):
        source = {
            key.removeprefix("backbone."): value
            for key, value in teacher.items()
            if key.startswith("backbone.") and isinstance(value, torch.Tensor)
        }
    else:
        source = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(key, str) and isinstance(value, torch.Tensor)
        }
    source = {
        key.replace("mlp.w12.", "mlp.fc1.").replace("mlp.w3.", "mlp.fc2."): value
        for key, value in source.items()
    }
    if "pos_embed" in source:
        source["pos_embed"] = _interpolate_pos_embed(
            source["pos_embed"], expected["pos_embed"].shape
        )

    missing = [key for key in expected if key not in source]
    mismatched = [
        (key, tuple(source[key].shape), tuple(expected[key].shape))
        for key in expected
        if key in source and source[key].shape != expected[key].shape
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"RETFound checkpoint is incompatible with {ARCHITECTURE}: "
            f"missing={missing}, shape_mismatch={mismatched}"
        )
    return {key: source[key] for key in expected}


def _replace_linear(
    block: nn.Module, path: str, config: BackboneConfig
) -> None:
    parent = block
    components = path.split(".")
    for component in components[:-1]:
        parent = getattr(parent, component)
    name = components[-1]
    layer = getattr(parent, name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"LoRA target {path} is not nn.Linear")
    setattr(
        parent,
        name,
        LoRALinear(
            layer,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        ),
    )


def _add_lora(backbone: nn.Module, config: BackboneConfig) -> None:
    blocks = backbone.blocks
    if not 1 <= config.lora_last_n_blocks <= len(blocks):
        raise ValueError(
            f"model.lora_last_n_blocks must be between 1 and {len(blocks)}"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for block in blocks[-config.lora_last_n_blocks:]:
        for target in LORA_TARGETS:
            _replace_linear(block, target, config)


def build_retfound_dinov2(config: BackboneConfig) -> BuiltBackbone:
    backbone = _create_model(config, meta=config.pretrained)
    if config.pretrained:
        state = _load_backbone_state(config, backbone.state_dict())
        backbone.load_state_dict(state, strict=True, assign=True)
    _add_lora(backbone, config)
    return BuiltBackbone(backbone, int(backbone.num_features))
