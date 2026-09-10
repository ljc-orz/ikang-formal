"""ConvNeXt backbone construction and V1 fine-tuning policy."""

from __future__ import annotations

import timm

from .base import BackboneConfig, BuiltBackbone, resolve_pretrained_weights


def build_convnext(config: BackboneConfig) -> BuiltBackbone:
    pretrained_cfg_overlay = None
    if config.pretrained:
        if config.pretrained_weights is None:
            raise ValueError(
                "pretrained ConvNeXt requires model.pretrained_weights"
            )
        weights_path = resolve_pretrained_weights(config.pretrained_weights)
        pretrained_cfg_overlay = {"file": str(weights_path)}

    backbone = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        num_classes=0,
        global_pool="avg",
        drop_path_rate=config.drop_path_rate,
    )
    if not hasattr(backbone, "stages") or len(backbone.stages) < 2:
        raise ValueError(
            "the convnext backbone type requires a model with at least two "
            f"stages: {config.model_name}"
        )

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for stage in backbone.stages[-2:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True
    if hasattr(backbone, "head"):
        for parameter in backbone.head.parameters():
            parameter.requires_grad = True
    return BuiltBackbone(backbone, int(backbone.num_features))
