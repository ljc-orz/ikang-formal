"""Model definitions and pluggable backbone registry."""

from .backbones import LoRALinear, available_backbones
from .fundus_classifier import FundusClassifier

__all__ = ["FundusClassifier", "LoRALinear", "available_backbones"]
