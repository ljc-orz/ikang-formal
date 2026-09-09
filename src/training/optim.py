"""Optimizer and cosine scheduler construction."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("backbone", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "backbone" if name.startswith("backbone.") else "head"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(scope, use_decay)].append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    for (scope, use_decay), parameters in groups.items():
        if parameters:
            parameter_groups.append(
                {
                    "params": parameters,
                    "scope": scope,
                    "lr": backbone_lr if scope == "backbone" else head_lr,
                    "weight_decay": weight_decay if use_decay else 0.0,
                }
            )
    return torch.optim.AdamW(parameter_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])

        def multiplier(epoch: int, base_lr: float = base_lr) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            cosine_epochs = max(max_epochs - warmup_epochs, 1)
            progress = min(max(epoch - warmup_epochs + 1, 0) / cosine_epochs, 1.0)
            minimum = min(min_lr / base_lr, 1.0)
            return minimum + (1.0 - minimum) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        lambdas.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
