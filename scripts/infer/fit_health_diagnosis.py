#!/usr/bin/env python3
"""Fit, calibrate, and evaluate patient-level health diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.health_diagnosis import (  # noqa: E402
    AggregatedPredictions,
    HealthDiagnosticModel,
    calibrate_threshold,
    check_compatible,
    fit_health_diagnostic,
    load_abnormal_labels,
    load_aggregated_predictions,
)
from src.training import binary_metrics  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-file",
        type=Path,
        required=True,
        help="Aggregate training predictions; out-of-fold predictions are recommended",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        required=True,
        help="Aggregate internal-validation predictions used to choose the threshold",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Optional aggregate external-validation predictions",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Override source_parquet metadata for all input files",
    )
    parser.add_argument(
        "--healthy-quantile",
        type=float,
        default=0.95,
        help="Healthy-region empirical quantile (default: 0.95)",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength for logistic regression (default: 1)",
    )
    return parser.parse_args(argv)


def _valid_training_tensors(
    data: AggregatedPredictions, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = labels >= 0
    if not bool(valid.any()):
        raise ValueError(f"no complete health labels in {data.path}")
    return data.logits[:, valid], data.eye_differences[:, valid], labels[valid]


def _prediction_payload(
    model: HealthDiagnosticModel,
    data: AggregatedPredictions,
    labels: torch.Tensor,
    parquet: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    output = model.predict(data.logits, data.eye_differences)
    valid = labels >= 0
    metrics: dict[str, float] = {
        "patients": float(labels.numel()),
        "labeled_patients": float(valid.sum()),
        "healthy_patients": float((labels == 0).sum()),
        "abnormal_patients": float((labels == 1).sum()),
    }
    if bool(valid.any()):
        metrics.update(
            binary_metrics(
                labels[valid].numpy(),
                output.abnormal_probability[valid].numpy(),
                threshold=model.decision_threshold,
            )
        )
    payload = {
        "source_row": data.source_rows,
        "labels": labels,
        "valid_label": valid,
        "abnormal_probability": output.abnormal_probability,
        "prediction": output.prediction,
        "health_distance": output.health_distance,
        "outside_fraction": output.outside_fraction,
        "tta_instability": output.tta_instability,
        "mean_logits": output.mean_logits,
        "tta_standard_deviation": output.tta_standard_deviation,
        "mean_eye_difference": output.mean_eye_difference,
        "indicators": data.indicators,
        "targets": data.targets,
        "seeds": data.seeds,
        "eye": data.eye,
        "split": data.split,
        "source_parquet": str(parquet),
        "decision_threshold": model.decision_threshold,
    }
    return payload, metrics


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    fit_data = load_aggregated_predictions(args.fit_file)
    calibration_data = load_aggregated_predictions(args.calibration_file)
    check_compatible(fit_data, calibration_data)
    test_data = (
        None
        if args.test_file is None
        else load_aggregated_predictions(args.test_file)
    )
    if test_data is not None:
        check_compatible(fit_data, test_data)

    fit_labels, fit_parquet = load_abnormal_labels(fit_data, args.parquet)
    calibration_labels, calibration_parquet = load_abnormal_labels(
        calibration_data, args.parquet
    )
    test_labels: torch.Tensor | None = None
    test_parquet: Path | None = None
    if test_data is not None:
        test_labels, test_parquet = load_abnormal_labels(test_data, args.parquet)

    fit_logits, fit_eye_differences, valid_fit_labels = _valid_training_tensors(
        fit_data, fit_labels
    )
    model = fit_health_diagnostic(
        fit_logits,
        fit_eye_differences,
        valid_fit_labels,
        fit_data.indicators,
        healthy_quantile=args.healthy_quantile,
        logistic_c=args.logistic_c,
    )
    calibration_logits, calibration_eye_differences, valid_calibration_labels = (
        _valid_training_tensors(calibration_data, calibration_labels)
    )
    model = calibrate_threshold(
        model,
        calibration_logits,
        calibration_eye_differences,
        valid_calibration_labels,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_state = model.to_state_dict()
    model_state.update(
        {
            "fit_file": str(fit_data.path),
            "calibration_file": str(calibration_data.path),
            "test_file": None if test_data is None else str(test_data.path),
        }
    )
    _atomic_torch_save(model_state, output_dir / "health_diagnosis_model.pt")

    datasets = [
        ("fit", fit_data, fit_labels, fit_parquet),
        ("calibration", calibration_data, calibration_labels, calibration_parquet),
    ]
    if test_data is not None and test_labels is not None and test_parquet is not None:
        datasets.append(("test", test_data, test_labels, test_parquet))
    all_metrics: dict[str, dict[str, float]] = {}
    for name, data, labels, parquet in datasets:
        payload, metrics = _prediction_payload(model, data, labels, parquet)
        _atomic_torch_save(payload, output_dir / f"{name}_predictions.pt")
        all_metrics[name] = metrics
        print(
            f"{name}: patients={int(metrics['patients'])} "
            f"labeled={int(metrics['labeled_patients'])} "
            f"auroc={metrics.get('auroc', np.nan):.6f} "
            f"auprc={metrics.get('auprc', np.nan):.6f}",
            flush=True,
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved model and predictions to {output_dir}; "
        f"decision_threshold={model.decision_threshold:.6f}",
        flush=True,
    )
    return all_metrics


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
