"""Loading aggregate prediction tensors and patient-level health labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AggregatedPredictions:
    path: Path
    logits: torch.Tensor
    eye_differences: torch.Tensor
    source_rows: torch.Tensor
    seeds: torch.Tensor
    indicators: tuple[str, ...]
    targets: tuple[str, ...]
    eye: str
    split: str
    transform: str
    source_parquet: str | None
    source_parquets: dict[str, str | None]


def parse_source_parquets(
    indicators: Sequence[str],
    source_parquet: Any,
    source_parquets: Any,
    *,
    context: Path,
) -> dict[str, str | None]:
    """Read new per-indicator sources while accepting older aggregate files."""
    common = None if source_parquet is None else str(source_parquet)
    if source_parquets is None:
        return {indicator: common for indicator in indicators}
    if not isinstance(source_parquets, Mapping):
        raise ValueError(f"source_parquets must be a mapping in {context}")
    if set(source_parquets) != set(indicators):
        raise ValueError(
            f"source_parquets keys do not match indicators in {context}"
        )
    return {
        indicator: (
            None
            if source_parquets[indicator] is None
            else str(source_parquets[indicator])
        )
        for indicator in indicators
    }


def load_aggregated_predictions(path: Path) -> AggregatedPredictions:
    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"aggregate file must contain a dictionary: {resolved}")
    logits = payload.get("X")
    eye_differences = payload.get("left_right_abs_difference")
    source_rows = payload.get("source_row")
    seeds = payload.get("seeds")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError(f"missing X [seeds, patients, indicators] in {resolved}")
    if not isinstance(eye_differences, torch.Tensor) or eye_differences.shape != logits.shape:
        raise ValueError(f"missing matching left_right_abs_difference in {resolved}")
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.dtype != torch.int64
        or source_rows.shape != (logits.shape[1],)
    ):
        raise ValueError(f"invalid source_row in {resolved}")
    if (
        not isinstance(seeds, torch.Tensor)
        or seeds.dtype != torch.int64
        or seeds.shape != (logits.shape[0],)
    ):
        raise ValueError(f"invalid seeds in {resolved}")
    indicators = tuple(payload.get("indicators", ()))
    targets = tuple(payload.get("targets", ()))
    if len(indicators) != logits.shape[2] or len(targets) != logits.shape[2]:
        raise ValueError(f"indicator/target metadata does not match X in {resolved}")
    if source_rows.numel() != torch.unique(source_rows).numel():
        raise ValueError(f"duplicate source_row values in {resolved}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"non-finite X values in {resolved}")
    if not bool(torch.isfinite(eye_differences).all()):
        raise ValueError(f"non-finite eye differences in {resolved}")
    source_parquet_value = payload.get("source_parquet")
    source_parquets = parse_source_parquets(
        indicators,
        source_parquet_value,
        payload.get("source_parquets"),
        context=resolved,
    )
    return AggregatedPredictions(
        path=resolved,
        logits=logits.to(dtype=torch.float32),
        eye_differences=eye_differences.to(dtype=torch.float32),
        source_rows=source_rows,
        seeds=seeds,
        indicators=indicators,
        targets=targets,
        eye=str(payload.get("eye", "")),
        split=str(payload.get("split", "")),
        transform=str(payload.get("transform", "")),
        source_parquet=(
            None if source_parquet_value is None else str(source_parquet_value)
        ),
        source_parquets=source_parquets,
    )


def load_target_values(
    *,
    indicators: Sequence[str],
    targets: Sequence[str],
    source_rows: torch.Tensor | np.ndarray,
    source_parquet: str | None,
    source_parquets: Mapping[str, str | None],
    parquet: Path | None,
    context: Path,
) -> tuple[np.ndarray, dict[str, Path]]:
    """Load aligned target columns, possibly from one parquet per indicator."""
    import pandas as pd

    rows = (
        source_rows.detach().cpu().numpy()
        if isinstance(source_rows, torch.Tensor)
        else np.asarray(source_rows)
    )
    if parquet is not None:
        resolved_override = parquet.expanduser().resolve(strict=True)
        paths = {indicator: resolved_override for indicator in indicators}
    else:
        parsed = parse_source_parquets(
            indicators,
            source_parquet,
            source_parquets,
            context=context,
        )
        missing = [indicator for indicator, value in parsed.items() if value is None]
        if missing:
            raise ValueError(
                f"{context} has no source parquet for indicators {missing}; "
                "specify --parquet"
            )
        paths = {
            indicator: Path(parsed[indicator]).expanduser().resolve(strict=True)
            for indicator in indicators
        }

    values = np.empty((len(rows), len(targets)), dtype=np.float64)
    grouped: dict[Path, list[tuple[int, str]]] = {}
    for index, (indicator, target) in enumerate(zip(indicators, targets)):
        grouped.setdefault(paths[indicator], []).append((index, target))
    for path, members in grouped.items():
        columns = list(dict.fromkeys(target for _, target in members))
        frame = pd.read_parquet(path, columns=columns)
        if rows.size and (rows.min() < 0 or rows.max() >= len(frame)):
            raise ValueError(
                f"source_row is outside parquet row range [0, {len(frame)}) in "
                f"{context} for {path}"
            )
        selected = frame.iloc[rows]
        for index, target in members:
            values[:, index] = pd.to_numeric(
                selected[target], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
    return values, paths


def load_abnormal_labels(
    predictions: AggregatedPredictions,
    parquet: Path | None = None,
) -> tuple[torch.Tensor, dict[str, Path]]:
    """Return 1 for any abnormal target, 0 for all-zero, and -1 if incomplete."""
    numeric, sources = load_target_values(
        indicators=predictions.indicators,
        targets=predictions.targets,
        source_rows=predictions.source_rows,
        source_parquet=predictions.source_parquet,
        source_parquets=predictions.source_parquets,
        parquet=parquet,
        context=predictions.path,
    )
    valid = np.logical_or(numeric == 0, numeric == 1).all(axis=1)
    labels = np.full(len(predictions.source_rows), -1, dtype=np.int64)
    labels[valid] = (numeric[valid] == 1).any(axis=1).astype(np.int64)
    return torch.from_numpy(labels), sources


def check_compatible(
    reference: AggregatedPredictions,
    other: AggregatedPredictions,
) -> None:
    for name in ("indicators", "targets", "eye", "transform"):
        if getattr(reference, name) != getattr(other, name):
            raise ValueError(
                f"{name} differs between {reference.path} and {other.path}"
            )
    if not torch.equal(reference.seeds, other.seeds):
        raise ValueError(f"seeds differ between {reference.path} and {other.path}")
