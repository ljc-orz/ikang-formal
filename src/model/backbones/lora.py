"""Small dependency-free LoRA building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Apply a trainable low-rank update to a frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("model.lora_rank must be positive")
        if alpha <= 0:
            raise ValueError("model.lora_alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.lora_dropout must be in [0, 1)")

        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b)
        return self.base(value) + update * self.scaling
