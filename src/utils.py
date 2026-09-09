"""Shared serialization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON-compatible null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
