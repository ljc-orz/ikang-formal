"""Backbone registry used by the fundus classifier."""

from __future__ import annotations

from collections.abc import Callable

from .base import BackboneConfig, BuiltBackbone
from .convnext import build_convnext
from .resnet50 import build_resnet50
from .retfound_dinov2 import build_retfound_dinov2


BackboneBuilder = Callable[[BackboneConfig], BuiltBackbone]

_BUILDERS: dict[str, BackboneBuilder] = {
    "convnext": build_convnext,
    "retfound_dinov2": build_retfound_dinov2,
    "resnet50": build_resnet50,
}


def available_backbones() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def build_backbone(config: BackboneConfig) -> BuiltBackbone:
    try:
        builder = _BUILDERS[config.kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown backbone {config.kind!r}; available: {available_backbones()}"
        ) from exc
    return builder(config)
