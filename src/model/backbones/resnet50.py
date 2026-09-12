"""ResNet-50 construction and configurable partial fine-tuning."""

from __future__ import annotations

from torch import nn
import timm

from .base import BackboneConfig, BuiltBackbone, resolve_pretrained_weights


def _residual_blocks(backbone: nn.Module) -> list[nn.Module]:
    layers = []
    for name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(backbone, name, None)
        if not isinstance(layer, nn.Sequential):
            raise ValueError(
                "the resnet50 backbone type requires timm ResNet layer1-layer4: "
                f"{type(backbone).__name__}"
            )
        layers.extend(layer)
    return layers


def build_resnet50(config: BackboneConfig) -> BuiltBackbone:
    pretrained_cfg_overlay = None
    if config.pretrained:
        if config.pretrained_weights is None:
            raise ValueError("pretrained ResNet-50 requires model.pretrained_weights")
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
    blocks = _residual_blocks(backbone)
    count = config.trainable_last_n_blocks
    if not 1 <= count <= len(blocks):
        raise ValueError(
            "model.trainable_last_n_blocks must be between 1 and "
            f"{len(blocks)} for ResNet-50"
        )

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for block in blocks[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return BuiltBackbone(backbone, int(backbone.num_features))
