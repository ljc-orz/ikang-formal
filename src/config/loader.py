"""Small YAML configuration loader with strict recursive overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("v1.yaml")


def _merge(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> None:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"unknown configuration key: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"configuration section {path} must be a mapping")
            _merge(base[key], value, path)
        else:
            base[key] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the V1 defaults and optionally merge a user configuration."""
    config = deepcopy(_read_yaml(DEFAULT_CONFIG_PATH))
    if path is not None:
        override_path = Path(path).resolve(strict=True)
        _merge(config, _read_yaml(override_path))
    return config


def apply_config_overrides(
    config: dict[str, Any], assignments: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Apply strict dotted-path assignments whose values use YAML syntax."""
    for assignment in assignments:
        key, separator, raw_value = assignment.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(
                f"invalid configuration override {assignment!r}; expected KEY=VALUE"
            )
        parts = key.split(".")
        section: dict[str, Any] = config
        for index, part in enumerate(parts):
            path = ".".join(parts[: index + 1])
            if part not in section:
                raise ValueError(f"unknown configuration key: {path}")
            if index == len(parts) - 1:
                if isinstance(section[part], dict):
                    raise ValueError(
                        f"configuration override must target a value, not section {path}"
                    )
                parsed = yaml.safe_load(raw_value)
                if isinstance(section[part], float):
                    try:
                        parsed = float(parsed)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"configuration override {path} must be a number"
                        ) from exc
                section[part] = parsed
                break
            value = section[part]
            if not isinstance(value, dict):
                raise ValueError(f"configuration value {path} has no child keys")
            section = value
    return config
