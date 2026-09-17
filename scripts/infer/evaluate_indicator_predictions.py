#!/usr/bin/env python3
"""Evaluate per-indicator seeded inference results on internal/external sets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, confusion_matrix


# These paths may be filled in once and then the script can be run without arguments.
INTERNAL_PREDICTIONS_DIR: Path | None = None
EXTERNAL_PREDICTIONS_DIR: Path | None = None
OUTPUT_DIR: Path | None = None


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.infer.reduce_predictions import (  # noqa: E402
    InputLayout,
    discover_layout,
    parse_seed_selection,
)
from src.training import binary_metrics, select_youden_threshold  # noqa: E402


@dataclass(frozen=True)
class IndicatorPredictions:
    indicator: str
    target: str
    split: str
    source_rows: np.ndarray
    seed_probabilities: np.ndarray
    probabilities: np.ndarray
    seeds: tuple[int, ...]
    source_parquet: str | None
    checkpoint: str
    transform: str
    eye: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-predictions-dir",
        type=Path,
        default=INTERNAL_PREDICTIONS_DIR,
        help="Root containing internal-validation indicator subdirectories",
    )
    parser.add_argument(
        "--external-predictions-dir",
        type=Path,
        default=EXTERNAL_PREDICTIONS_DIR,
        help="Root containing external-validation indicator subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for indicator_metrics.csv/json",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        help="Indicator subdirectories (default: sorted automatic discovery)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        help="Seeds, or 'seq FIRST LAST' (default: all common seeds)",
    )
    parser.add_argument(
        "--eye",
        choices=("left", "right", "mean"),
        default="mean",
        help="Eye probability to evaluate; mean averages left/right probabilities",
    )
    parser.add_argument(
        "--internal-parquet",
        type=Path,
        help="Override source_parquet metadata for the internal set",
    )
    parser.add_argument(
        "--external-parquet",
        type=Path,
        help="Override source_parquet metadata for the external set",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1_000,
        help="TTA seed-bootstrap repetitions for 95%% CI (default: 1000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260917,
        help="Random seed for TTA bootstrap",
    )
    args = parser.parse_args(argv)
    try:
        args.seeds = parse_seed_selection(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    missing = [
        option
        for option, value in (
            ("--internal-predictions-dir", args.internal_predictions_dir),
            ("--external-predictions-dir", args.external_predictions_dir),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "set the constants at the top of the script or provide " + ", ".join(missing)
        )
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"prediction file must contain a dictionary: {path}")
    return payload


def _eye_probabilities(
    logits: torch.Tensor, eye_order: Sequence[str], eye: str, *, path: Path
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"{path}: logits must have shape [patients, 2]")
    order = tuple(eye_order)
    if len(order) != 2 or set(order) != {"left", "right"}:
        raise ValueError(f"{path}: invalid eye_order {order!r}")
    probabilities = logits.to(dtype=torch.float32).sigmoid()
    if eye == "mean":
        return probabilities.mean(dim=1)
    return probabilities[:, order.index(eye)]


def load_indicator_predictions(
    layout: InputLayout, indicator: str, eye: str
) -> IndicatorPredictions:
    canonical_rows: torch.Tensor | None = None
    seed_probabilities: list[torch.Tensor] = []
    reference_target: str | None = None
    reference_parquet: str | None = None
    reference_checkpoint: str | None = None
    reference_transform: str | None = None
    parquet_initialized = False
    for seed in layout.seeds:
        path = layout.files[indicator][seed]
        payload = _load_payload(path)
        if int(payload.get("seed", -1)) != seed or payload.get("split") != layout.split:
            raise ValueError(f"filename/metadata mismatch in {path}")
        source_rows = payload.get("source_row")
        logits = payload.get("logits")
        if not isinstance(source_rows, torch.Tensor) or source_rows.ndim != 1:
            raise ValueError(f"{path}: invalid source_row tensor")
        if source_rows.dtype != torch.int64:
            raise ValueError(f"{path}: source_row must be int64")
        if source_rows.numel() != torch.unique(source_rows).numel():
            raise ValueError(f"{path}: duplicate source_row values")
        if not isinstance(logits, torch.Tensor) or logits.shape[0] != len(source_rows):
            raise ValueError(f"{path}: source_row/logits length mismatch")
        order = torch.argsort(source_rows)
        sorted_rows = source_rows[order]
        if canonical_rows is None:
            canonical_rows = sorted_rows
        elif not torch.equal(canonical_rows, sorted_rows):
            raise ValueError(f"patient source_row set differs in {path}")
        values = _eye_probabilities(
            logits, payload.get("eye_order", ()), eye, path=path
        )[order]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"non-finite probabilities in {path}")
        seed_probabilities.append(values)

        target = str(payload.get("target", ""))
        checkpoint = str(payload.get("checkpoint", ""))
        if not target or not checkpoint:
            raise ValueError(f"missing target or checkpoint metadata in {path}")
        if reference_target is None:
            reference_target = target
            reference_checkpoint = checkpoint
        elif target != reference_target or checkpoint != reference_checkpoint:
            raise ValueError(f"target or checkpoint changed between seeds in {path}")
        transform = str(payload.get("transform", ""))
        if not transform:
            raise ValueError(f"missing transform metadata in {path}")
        if reference_transform is None:
            reference_transform = transform
        elif transform != reference_transform:
            raise ValueError(f"transform changed between seeds in {path}")
        parquet_value = payload.get("source_parquet")
        parquet = None if parquet_value is None else str(parquet_value)
        if not parquet_initialized:
            reference_parquet = parquet
            parquet_initialized = True
        elif parquet != reference_parquet:
            raise ValueError(f"source_parquet changed between seeds in {path}")

    if (
        canonical_rows is None
        or not seed_probabilities
        or reference_target is None
        or reference_checkpoint is None
        or reference_transform is None
    ):
        raise RuntimeError(f"no predictions loaded for {indicator}")
    stacked_probabilities = torch.stack(seed_probabilities, dim=0)
    return IndicatorPredictions(
        indicator=indicator,
        target=reference_target,
        split=layout.split,
        source_rows=canonical_rows.numpy(),
        seed_probabilities=stacked_probabilities.numpy(),
        probabilities=stacked_probabilities.mean(dim=0).numpy(),
        seeds=layout.seeds,
        source_parquet=reference_parquet,
        checkpoint=reference_checkpoint,
        transform=reference_transform,
        eye=eye,
    )


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return value


def metric_row(
    data: IndicatorPredictions,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    valid = np.isin(labels, (0, 1))
    valid_labels = labels[valid].astype(np.int64, copy=False)
    valid_probabilities = data.probabilities[valid].astype(np.float64, copy=False)
    if not valid.any():
        raise ValueError(f"{data.indicator}/{data.split}: no labels equal to 0 or 1")
    if len(np.unique(valid_labels)) != 2:
        raise ValueError(
            f"{data.indicator}/{data.split}: AUROC/AUPRC require both label classes"
        )
    values = performance_values(valid_labels, valid_probabilities, threshold)
    predictions = (valid_probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        valid_labels, predictions, labels=(0, 1)
    ).ravel()
    return {
        "split": data.split,
        "indicator": data.indicator,
        "target": data.target,
        "patients": len(labels),
        "labeled_patients": int(valid.sum()),
        "negative_patients": int((valid_labels == 0).sum()),
        "positive_patients": int((valid_labels == 1).sum()),
        "prevalence": float(valid_labels.mean()),
        "seed_count": len(data.seeds),
        "eye": data.eye,
        "transform": data.transform,
        "checkpoint": data.checkpoint,
        "threshold": float(threshold),
        **values,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_probability": float(valid_probabilities.mean()),
    }


CI_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "ece",
    "sensitivity",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def performance_values(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    basic = binary_metrics(labels, probabilities, threshold=threshold)
    predictions = (probabilities >= threshold).astype(np.int64)
    accuracy = float((predictions == labels).mean())
    sensitivity = basic["sensitivity"]
    specificity = basic["specificity"]
    return {
        "auroc": basic["auroc"],
        "auprc": basic["auprc"],
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
    }


def tta_bootstrap_intervals(
    data: IndicatorPredictions,
    labels: np.ndarray,
    *,
    threshold: float,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap TTA seeds while keeping the evaluated patient cohort fixed."""
    if len(data.seeds) < 2:
        raise ValueError(
            f"{data.indicator}/{data.split}: TTA confidence intervals require "
            "at least two seeds"
        )
    valid = np.isin(labels, (0, 1))
    valid_labels = labels[valid].astype(np.int64, copy=False)
    probabilities = data.seed_probabilities[:, valid].astype(np.float64, copy=False)
    distributions = {
        metric: np.empty(samples, dtype=np.float64) for metric in CI_METRICS
    }
    seed_count = probabilities.shape[0]
    for iteration in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_mean = probabilities[sampled_seeds].mean(axis=0)
        values = performance_values(valid_labels, sampled_mean, threshold)
        for metric in CI_METRICS:
            distributions[metric][iteration] = values[metric]
    intervals: dict[str, float] = {}
    for metric, values in distributions.items():
        low, high = np.quantile(values, (0.025, 0.975))
        intervals[f"{metric}_ci_low"] = float(low)
        intervals[f"{metric}_ci_high"] = float(high)
    return intervals


def _resolve_parquet(data: IndicatorPredictions, override: Path | None) -> Path:
    if override is not None:
        return override.resolve(strict=True)
    if data.source_parquet is None:
        raise ValueError(
            f"{data.indicator}/{data.split} has no source_parquet; provide an override"
        )
    return Path(data.source_parquet).expanduser().resolve(strict=True)


def load_labels(
    datasets: Sequence[IndicatorPredictions], override: Path | None
) -> dict[str, np.ndarray]:
    grouped: dict[Path, list[IndicatorPredictions]] = {}
    for data in datasets:
        grouped.setdefault(_resolve_parquet(data, override), []).append(data)
    labels: dict[str, np.ndarray] = {}
    for path, members in grouped.items():
        targets = sorted({member.target for member in members})
        frame = pd.read_parquet(path, columns=targets)
        for member in members:
            if member.source_rows.size and (
                member.source_rows.min() < 0 or member.source_rows.max() >= len(frame)
            ):
                raise ValueError(f"source_row is outside parquet bounds for {member.indicator}")
            values = pd.to_numeric(
                frame.iloc[member.source_rows][member.target], errors="coerce"
            ).to_numpy()
            labels[member.indicator] = values
    return labels


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    internal_layout = discover_layout(
        args.internal_predictions_dir,
        indicators=args.indicators,
        split="internal_validation",
        seeds=args.seeds,
    )
    external_layout = discover_layout(
        args.external_predictions_dir,
        indicators=internal_layout.indicators,
        split="external_validation",
        seeds=args.seeds,
    )
    if internal_layout.seeds != external_layout.seeds:
        raise ValueError(
            "internal/external seed sets differ; specify --seeds to select a common set"
        )
    internal = {
        indicator: load_indicator_predictions(internal_layout, indicator, args.eye)
        for indicator in internal_layout.indicators
    }
    external = {
        indicator: load_indicator_predictions(external_layout, indicator, args.eye)
        for indicator in external_layout.indicators
    }
    for indicator in internal_layout.indicators:
        if internal[indicator].target != external[indicator].target:
            raise ValueError(f"target differs between datasets for {indicator}")
        if internal[indicator].checkpoint != external[indicator].checkpoint:
            raise ValueError(f"checkpoint differs between datasets for {indicator}")
        if internal[indicator].transform != external[indicator].transform:
            raise ValueError(f"transform differs between datasets for {indicator}")

    internal_labels = load_labels(list(internal.values()), args.internal_parquet)
    external_labels = load_labels(list(external.values()), args.external_parquet)
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for indicator_index, indicator in enumerate(internal_layout.indicators, start=1):
        print(
            f"[{indicator_index}/{len(internal_layout.indicators)}] "
            f"evaluating {indicator} with {args.bootstrap_samples} TTA bootstraps",
            flush=True,
        )
        labels = internal_labels[indicator]
        valid = np.isin(labels, (0, 1))
        if not valid.any():
            raise ValueError(f"{indicator}/internal_validation has no binary labels")
        thresholds[indicator] = select_youden_threshold(
            labels[valid].astype(np.int64), internal[indicator].probabilities[valid]
        )
        internal_row = metric_row(
            internal[indicator], labels, threshold=thresholds[indicator]
        )
        internal_row.update(
            tta_bootstrap_intervals(
                internal[indicator],
                labels,
                threshold=thresholds[indicator],
                samples=args.bootstrap_samples,
                rng=rng,
            )
        )
        rows.append(internal_row)
        external_row = metric_row(
            external[indicator],
            external_labels[indicator],
            threshold=thresholds[indicator],
        )
        external_row.update(
            tta_bootstrap_intervals(
                external[indicator],
                external_labels[indicator],
                threshold=thresholds[indicator],
                samples=args.bootstrap_samples,
                rng=rng,
            )
        )
        rows.append(external_row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "indicator_metrics.csv", rows)
    summary = {
        "probability_aggregation": "mean sigmoid probability across selected seeds and eyes",
        "threshold_source": "internal_validation Youden index",
        "confidence_interval": {
            "level": 0.95,
            "method": "nonparametric bootstrap over TTA seeds",
            "patient_cohort": "fixed",
            "samples": args.bootstrap_samples,
            "random_seed": args.bootstrap_seed,
        },
        "eye": args.eye,
        "seeds": list(internal_layout.seeds),
        "indicators": list(internal_layout.indicators),
        "metrics": rows,
    }
    (output_dir / "indicator_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "split                 indicator      AUROC (TTA 95% CI)"
        "                   AUPRC (TTA 95% CI)",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['split']:<21} {row['indicator']:<12} "
            f"{row['auroc']:.4f} [{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}]  "
            f"{row['auprc']:.4f} [{row['auprc_ci_low']:.4f}, {row['auprc_ci_high']:.4f}]",
            flush=True,
        )
    print(f"saved per-indicator metrics to {output_dir}", flush=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
