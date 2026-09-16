"""Loading aggregate prediction tensors and patient-level health labels."""

from __future__ import annotations

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
    )


def load_abnormal_labels(
    predictions: AggregatedPredictions,
    parquet: Path | None = None,
) -> tuple[torch.Tensor, Path]:
    """Return 1 for any abnormal target, 0 for all-zero, and -1 if incomplete."""
    import pandas as pd

    if parquet is None:
        if predictions.source_parquet is None:
            raise ValueError(
                f"{predictions.path} has no source_parquet; specify --parquet"
            )
        parquet_path = Path(predictions.source_parquet)
    else:
        parquet_path = parquet
    parquet_path = parquet_path.resolve(strict=True)
    frame = pd.read_parquet(parquet_path, columns=list(predictions.targets))
    rows = predictions.source_rows.numpy()
    if rows.size and (rows.min() < 0 or rows.max() >= len(frame)):
        raise ValueError(
            f"source_row is outside parquet row range [0, {len(frame)}) in "
            f"{predictions.path}"
        )
    numeric = frame.iloc[rows].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.logical_or(numeric == 0, numeric == 1).all(axis=1)
    labels = np.full(len(rows), -1, dtype=np.int64)
    labels[valid] = (numeric[valid] == 1).any(axis=1).astype(np.int64)
    return torch.from_numpy(labels), parquet_path


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
