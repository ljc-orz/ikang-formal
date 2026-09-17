#!/usr/bin/env python3
"""Visualize and compare patient-level health diagnosis results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.training import select_youden_threshold  # noqa: E402


SPLITS = ("fit", "calibration", "test")
SPLIT_TITLES = {
    "fit": "Fit",
    "calibration": "Calibration",
    "test": "External test",
}
MODEL_TITLES = {"resnet": "ResNet", "retfound": "RETFound"}
DEFAULT_COLORS = ("#4C78A8", "#F58518", "#54A24B")
CLASS_COLORS = {0: "#4C78A8", 1: "#E45756"}


@dataclass(frozen=True)
class PredictionData:
    source_rows: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    health_distance: np.ndarray
    outside_fraction: np.ndarray
    instability: np.ndarray
    threshold: float


@dataclass(frozen=True)
class ModelData:
    name: str
    title: str
    root: Path
    splits: dict[str, PredictionData]
    feature_names: tuple[str, ...]
    classifier_weights: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_DIR / "example_results",
        help="Root containing <model>/diagnosis directories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("resnet", "retfound"),
        help="Exactly two model directory names (default: resnet retfound)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <results-root>/diagnosis_analysis",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1_000,
        help="Stratified paired bootstrap repetitions (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260917,
        help="Bootstrap random seed",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def display_name(name: str) -> str:
    return MODEL_TITLES.get(name.lower(), name.replace("_", " ").title())


def _as_numpy(value: Any, *, path: Path, key: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: {key} must be a tensor")
    return value.detach().cpu().numpy()


def load_prediction(path: Path) -> PredictionData:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"prediction file must contain a dictionary: {path}")
    source_rows = _as_numpy(payload.get("source_row"), path=path, key="source_row")
    labels = _as_numpy(payload.get("labels"), path=path, key="labels")
    probabilities = _as_numpy(
        payload.get("abnormal_probability"), path=path, key="abnormal_probability"
    )
    predictions = _as_numpy(payload.get("prediction"), path=path, key="prediction")
    health_distance = _as_numpy(
        payload.get("health_distance"), path=path, key="health_distance"
    )
    outside_fraction = _as_numpy(
        payload.get("outside_fraction"), path=path, key="outside_fraction"
    )
    instability = _as_numpy(
        payload.get("tta_instability"), path=path, key="tta_instability"
    )
    valid_value = payload.get("valid_label")
    valid = (
        _as_numpy(valid_value, path=path, key="valid_label").astype(bool)
        if valid_value is not None
        else labels >= 0
    )
    arrays = (
        source_rows,
        labels,
        probabilities,
        predictions,
        health_distance,
        outside_fraction,
        instability,
        valid,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(f"all patient fields must be one-dimensional: {path}")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError(f"patient field lengths differ: {path}")
    valid &= labels >= 0
    if not bool(valid.any()):
        raise ValueError(f"no labeled patients in {path}")
    source_rows = source_rows[valid].astype(np.int64, copy=False)
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError(f"duplicate source_row values in {path}")
    order = np.argsort(source_rows)
    labels = labels[valid][order].astype(np.int64, copy=False)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"labels must be binary in {path}")
    threshold = float(payload.get("decision_threshold", math.nan))
    if not math.isfinite(threshold):
        raise ValueError(f"missing decision_threshold in {path}")
    data = PredictionData(
        source_rows=source_rows[order],
        labels=labels,
        probabilities=probabilities[valid][order].astype(np.float64, copy=False),
        predictions=predictions[valid][order].astype(np.int64, copy=False),
        health_distance=health_distance[valid][order].astype(np.float64, copy=False),
        outside_fraction=outside_fraction[valid][order].astype(np.float64, copy=False),
        instability=instability[valid][order].astype(np.float64, copy=False),
        threshold=threshold,
    )
    numeric_fields = (
        data.probabilities,
        data.health_distance,
        data.outside_fraction,
        data.instability,
    )
    if not all(np.isfinite(values).all() for values in numeric_fields):
        raise ValueError(f"non-finite patient values in {path}")
    return data


def load_model(results_root: Path, name: str) -> ModelData:
    root = (results_root / name / "diagnosis").resolve(strict=True)
    splits = {
        split: load_prediction(root / f"{split}_predictions.pt")
        for split in SPLITS
    }
    state_path = root / "health_diagnosis_model.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    feature_names = tuple(state.get("feature_names", ()))
    classifier_weights = _as_numpy(
        state.get("classifier_weight"), path=state_path, key="classifier_weight"
    ).astype(np.float64, copy=False)
    if len(feature_names) != len(classifier_weights):
        raise ValueError(f"feature names/weights differ in {state_path}")
    return ModelData(
        name=name,
        title=display_name(name),
        root=root,
        splits=splits,
        feature_names=feature_names,
        classifier_weights=classifier_weights,
    )


def validate_alignment(models: Sequence[ModelData]) -> None:
    reference = models[0]
    for split in SPLITS:
        expected = reference.splits[split]
        for model in models[1:]:
            current = model.splits[split]
            if not np.array_equal(expected.source_rows, current.source_rows):
                raise ValueError(
                    f"source_row set differs for {split}: "
                    f"{reference.title} vs {model.title}"
                )
            if not np.array_equal(expected.labels, current.labels):
                raise ValueError(
                    f"labels differ for {split}: {reference.title} vs {model.title}"
                )


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if mask.any():
            value += float(mask.sum()) / total * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return value


def metric_values(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=(0, 1)).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    return {
        "patients": float(len(labels)),
        "healthy_patients": float((labels == 0).sum()),
        "abnormal_patients": float((labels == 1).sum()),
        "prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def bootstrap_split(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], tuple[float, float]], list[dict[str, Any]]]:
    distributions = {
        (model, metric): np.empty(samples, dtype=np.float64)
        for model in probabilities
        for metric in ("auroc", "auprc")
    }
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    for iteration in range(samples):
        sample_indices = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        sample_labels = labels[sample_indices]
        for model, scores in probabilities.items():
            sample_scores = scores[sample_indices]
            distributions[(model, "auroc")][iteration] = roc_auc_score(
                sample_labels, sample_scores
            )
            distributions[(model, "auprc")][iteration] = average_precision_score(
                sample_labels, sample_scores
            )
    intervals = {
        key: tuple(np.quantile(values, (0.025, 0.975)).tolist())
        for key, values in distributions.items()
    }
    model_names = list(probabilities)
    ensemble = "ensemble"
    pairs = [(ensemble, model) for model in model_names if model != ensemble]
    base_models = [name for name in model_names if name != ensemble]
    if len(base_models) == 2:
        pairs.append((base_models[0], base_models[1]))
    comparisons: list[dict[str, Any]] = []
    for first, second in pairs:
        for metric in ("auroc", "auprc"):
            difference = distributions[(first, metric)] - distributions[(second, metric)]
            low, high = np.quantile(difference, (0.025, 0.975))
            comparisons.append(
                {
                    "first": first,
                    "second": second,
                    "metric": metric,
                    "difference": float(
                        (roc_auc_score if metric == "auroc" else average_precision_score)(
                            labels, probabilities[first]
                        )
                        - (roc_auc_score if metric == "auroc" else average_precision_score)(
                            labels, probabilities[second]
                        )
                    ),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "two_sided_sign_p": float(
                        min(np.mean(difference <= 0), np.mean(difference >= 0)) * 2
                    ),
                }
            )
    return intervals, comparisons


def reliability_points(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    means: list[float] = []
    fractions: list[float] = []
    counts: list[int] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if mask.any():
            means.append(float(probabilities[mask].mean()))
            fractions.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(means), np.asarray(fractions), np.asarray(counts)


def instability_analysis(data: PredictionData) -> dict[str, Any]:
    errors = (data.predictions != data.labels).astype(np.float64)
    correlation = float(spearmanr(data.instability, errors).statistic)
    quantiles = np.quantile(data.instability, np.linspace(0.0, 1.0, 6))
    quintile_error: list[float] = []
    for index in range(5):
        mask = (data.instability >= quantiles[index]) & (
            data.instability <= quantiles[index + 1]
            if index == 4
            else data.instability < quantiles[index + 1]
        )
        quintile_error.append(float(errors[mask].mean()) if mask.any() else math.nan)
    order = np.argsort(data.instability)
    coverages = np.linspace(0.5, 1.0, 11)
    selective_error: list[float] = []
    selective_auroc: list[float] = []
    for coverage in coverages:
        keep = order[: max(1, int(len(order) * coverage))]
        selective_error.append(float(errors[keep].mean()))
        selective_auroc.append(
            float(roc_auc_score(data.labels[keep], data.probabilities[keep]))
            if len(np.unique(data.labels[keep])) == 2
            else math.nan
        )
    return {
        "error_spearman": correlation,
        "quintile_error": quintile_error,
        "coverages": coverages.tolist(),
        "selective_error": selective_error,
        "selective_auroc": selective_auroc,
    }


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_performance(
    metric_rows: list[dict[str, Any]], model_order: Sequence[str], path: Path, dpi: int
) -> None:
    metrics = ("auroc", "auprc", "sensitivity", "specificity")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    x = np.arange(len(SPLITS))
    width = 0.24
    colors = dict(zip(model_order, DEFAULT_COLORS))
    for axis, metric in zip(axes.flat, metrics):
        for model_index, model in enumerate(model_order):
            values = [
                next(
                    row[metric]
                    for row in metric_rows
                    if row["split"] == split and row["model"] == model
                )
                for split in SPLITS
            ]
            axis.bar(
                x + (model_index - (len(model_order) - 1) / 2) * width,
                values,
                width,
                label=display_name(model),
                color=colors[model],
            )
        axis.set_title(metric.upper())
        axis.set_xticks(x, [SPLIT_TITLES[split] for split in SPLITS])
        axis.set_ylim(0.45, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="lower left", frameon=False)
    fig.suptitle("Diagnosis performance across datasets", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_roc_pr(
    labels: dict[str, np.ndarray],
    probabilities: dict[str, dict[str, np.ndarray]],
    model_order: Sequence[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    colors = dict(zip(model_order, DEFAULT_COLORS))
    for column, split in enumerate(SPLITS):
        y = labels[split]
        for model in model_order:
            scores = probabilities[split][model]
            fpr, tpr, _ = roc_curve(y, scores)
            precision, recall, _ = precision_recall_curve(y, scores)
            axes[0, column].plot(
                fpr,
                tpr,
                color=colors[model],
                label=f"{display_name(model)} ({roc_auc_score(y, scores):.3f})",
            )
            axes[1, column].plot(
                recall,
                precision,
                color=colors[model],
                label=f"{display_name(model)} ({average_precision_score(y, scores):.3f})",
            )
        axes[0, column].plot((0, 1), (0, 1), "--", color="0.65")
        axes[1, column].axhline(y.mean(), linestyle="--", color="0.65")
        axes[0, column].set_title(SPLIT_TITLES[split])
        axes[0, column].set_xlabel("False-positive rate")
        axes[0, column].set_ylabel("True-positive rate")
        axes[1, column].set_xlabel("Recall")
        axes[1, column].set_ylabel("Precision")
        axes[0, column].legend(frameon=False, fontsize=9)
        axes[1, column].legend(frameon=False, fontsize=9)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    fig.suptitle("ROC and precision-recall curves", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_confusions(
    labels: dict[str, np.ndarray],
    probabilities: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    model_order: Sequence[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, len(model_order), figsize=(4.5 * len(model_order), 12))
    for row, split in enumerate(SPLITS):
        for column, model in enumerate(model_order):
            matrix = confusion_matrix(
                labels[split],
                probabilities[split][model] >= thresholds[model],
                labels=(0, 1),
            )
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            axis = axes[row, column]
            image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for y_index in range(2):
                for x_index in range(2):
                    axis.text(
                        x_index,
                        y_index,
                        f"{normalized[y_index, x_index]:.1%}\n(n={matrix[y_index, x_index]})",
                        ha="center",
                        va="center",
                        color="white" if normalized[y_index, x_index] > 0.55 else "black",
                    )
            axis.set_xticks((0, 1), ("Healthy", "Abnormal"))
            axis.set_yticks((0, 1), ("Healthy", "Abnormal"))
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(f"{SPLIT_TITLES[split]} — {display_name(model)}")
    fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    fig.suptitle("Row-normalized confusion matrices", fontsize=16)
    fig.subplots_adjust(top=0.93, hspace=0.38, wspace=0.25)
    save_figure(fig, path, dpi)


def plot_probability_distributions(
    labels: dict[str, np.ndarray],
    probabilities: dict[str, dict[str, np.ndarray]],
    model_order: Sequence[str],
    path: Path,
    dpi: int,
    *,
    splits: Sequence[str] = SPLITS,
    title: str = "Abnormal-score distributions",
    split_titles: dict[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(
        len(splits),
        len(model_order),
        figsize=(4.8 * len(model_order), 3.7 * len(splits)),
        squeeze=False,
    )
    bins = np.linspace(0.0, 1.0, 31)
    for row, split in enumerate(splits):
        for column, model in enumerate(model_order):
            axis = axes[row, column]
            for label, class_title in ((0, "Healthy"), (1, "Abnormal")):
                axis.hist(
                    probabilities[split][model][labels[split] == label],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=CLASS_COLORS[label],
                    label=class_title,
                )
            axis.set_xlim(0, 1)
            split_title = (
                SPLIT_TITLES[split]
                if split_titles is None
                else split_titles.get(split, SPLIT_TITLES[split])
            )
            axis.set_title(f"{split_title} — {display_name(model)}")
            axis.set_xlabel("Abnormal score")
            axis.set_ylabel("Density")
            axis.grid(alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_calibration(
    labels: dict[str, np.ndarray],
    probabilities: dict[str, dict[str, np.ndarray]],
    model_order: Sequence[str],
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    colors = dict(zip(model_order, DEFAULT_COLORS))
    for axis, split in zip(axes, SPLITS):
        axis.plot((0, 1), (0, 1), "--", color="0.55", label="Ideal")
        for model in model_order:
            means, fractions, counts = reliability_points(
                labels[split], probabilities[split][model]
            )
            axis.plot(
                means,
                fractions,
                marker="o",
                color=colors[model],
                label=f"{display_name(model)} (ECE={expected_calibration_error(labels[split], probabilities[split][model]):.3f})",
            )
            axis.scatter(means, fractions, s=np.sqrt(counts) * 2, color=colors[model])
        axis.set_title(SPLIT_TITLES[split])
        axis.set_xlabel("Mean abnormal score")
        axis.set_ylabel("Observed abnormal fraction")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Reliability diagrams", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_health_distance(
    models: Sequence[ModelData], path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.5 * len(models)))
    for row, model in enumerate(models):
        for column, split in enumerate(SPLITS):
            axis = axes[row, column]
            data = model.splits[split]
            upper = float(np.quantile(data.health_distance, 0.995))
            bins = np.linspace(0.0, upper, 35)
            for label, title in ((0, "Healthy"), (1, "Abnormal")):
                axis.hist(
                    np.clip(data.health_distance[data.labels == label], 0, upper),
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=CLASS_COLORS[label],
                    label=title,
                )
            correlation = spearmanr(data.health_distance, data.probabilities).statistic
            axis.set_title(
                f"{model.title} — {SPLIT_TITLES[split]}\nscore correlation={correlation:.3f}"
            )
            axis.set_xlabel("Health distance (clipped at P99.5)")
            axis.set_ylabel("Density")
            axis.grid(alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle("Health-distance distributions", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_stability(
    models: Sequence[ModelData], analyses: dict[str, dict[str, Any]], path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = dict(zip((model.name for model in models), DEFAULT_COLORS))
    for column, split in enumerate(SPLITS):
        for model in models:
            result = analyses[split][model.name]
            axes[0, column].plot(
                np.arange(1, 6),
                result["quintile_error"],
                marker="o",
                color=colors[model.name],
                label=model.title,
            )
            axes[1, column].plot(
                result["coverages"],
                result["selective_error"],
                marker="o",
                color=colors[model.name],
                label=model.title,
            )
        axes[0, column].set_title(SPLIT_TITLES[split])
        axes[0, column].set_xlabel("Instability quintile (low to high)")
        axes[0, column].set_ylabel("Classification error")
        axes[1, column].set_xlabel("Coverage after retaining least-unstable patients")
        axes[1, column].set_ylabel("Classification error")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[0, column].legend(frameon=False)
        axes[1, column].legend(frameon=False)
    fig.suptitle("TTA instability and selective risk", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_agreement(
    models: Sequence[ModelData], path: Path, dpi: int
) -> None:
    first, second = models
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, split in zip(axes, SPLITS):
        first_data = first.splits[split]
        second_data = second.splits[split]
        image = axis.hexbin(
            first_data.probabilities,
            second_data.probabilities,
            gridsize=45,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        axis.plot((0, 1), (0, 1), "--", color="white", linewidth=1)
        axis.axvline(first_data.threshold, color="#A8DADC", linewidth=1)
        axis.axhline(second_data.threshold, color="#A8DADC", linewidth=1)
        agreement = float(np.mean(first_data.predictions == second_data.predictions))
        correlation = float(
            spearmanr(first_data.probabilities, second_data.probabilities).statistic
        )
        axis.set_title(
            f"{SPLIT_TITLES[split]}\nagreement={agreement:.1%}, Spearman={correlation:.3f}"
        )
        axis.set_xlabel(f"{first.title} abnormal score")
        axis.set_ylabel(f"{second.title} abnormal score")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        fig.colorbar(image, ax=axis, label="log10(count)")
    fig.suptitle("Paired model agreement", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_coefficients(models: Sequence[ModelData], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 7))
    if len(models) == 1:
        axes = np.asarray([axes])
    for axis, model in zip(axes, models):
        count = min(15, len(model.classifier_weights))
        selected = np.argsort(np.abs(model.classifier_weights))[-count:]
        selected = selected[np.argsort(model.classifier_weights[selected])]
        values = model.classifier_weights[selected]
        names = [model.feature_names[index] for index in selected]
        colors = ["#E45756" if value > 0 else "#4C78A8" for value in values]
        axis.barh(np.arange(count), values, color=colors)
        axis.set_yticks(np.arange(count), names, fontsize=8)
        axis.axvline(0, color="0.35", linewidth=1)
        axis.set_xlabel("Standardized logistic coefficient")
        axis.set_title(model.title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Largest diagnosis-model coefficients", fontsize=16)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_ci(row: dict[str, Any], metric: str) -> str:
    return f"{row[metric]:.4f} [{row[f'{metric}_ci_low']:.4f}, {row[f'{metric}_ci_high']:.4f}]"


def write_report(
    path: Path,
    *,
    models: Sequence[ModelData],
    metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    stability: dict[str, dict[str, Any]],
    agreement: dict[str, dict[str, float]],
) -> None:
    by_key = {(row["split"], row["model"]): row for row in metric_rows}
    test_rows = [by_key[("test", model.name)] for model in models]
    ensemble_test = by_key[("test", "ensemble")]
    ensemble_deltas = {
        (row["second"], row["metric"]): row
        for row in comparison_rows
        if row["split"] == "test" and row["first"] == "ensemble"
    }
    lines = [
        "# ResNet / RETFound 健康诊断结果分析",
        "",
        "## 数据与方法",
        "",
        "本报告直接读取两种骨干的 `diagnosis` 输出，按 `source_row` 对齐患者。"
        "平均集成使用两个异常评分的算术平均，阈值只在内部校准集上通过 Youden index 选择，"
        "随后固定应用于训练集和外部测试集。AUROC/AUPRC 区间及模型差值采用患者级、"
        "按健康状态分层的配对 bootstrap。",
        "",
        "## 外部测试集主要结果",
        "",
        "| Model | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity | Specificity | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows + [ensemble_test]:
        lines.append(
            f"| {display_name(row['model'])} | {_format_ci(row, 'auroc')} | "
            f"{_format_ci(row, 'auprc')} | {row['sensitivity']:.4f} | "
            f"{row['specificity']:.4f} | {row['brier']:.4f} | {row['ece']:.4f} |"
        )
    first, second = models
    first_test, second_test = test_rows
    first_auc_delta = ensemble_deltas[(first.name, "auroc")]
    second_auc_delta = ensemble_deltas[(second.name, "auroc")]
    first_ap_delta = ensemble_deltas[(first.name, "auprc")]
    second_ap_delta = ensemble_deltas[(second.name, "auprc")]
    external_agreement = agreement["test"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 两个单模型的外部区分性能接近：{first.title} AUROC={first_test['auroc']:.4f}，"
            f"{second.title} AUROC={second_test['auroc']:.4f}。两者差值的 bootstrap CI 跨0，"
            "当前数据不足以判定某一个骨干在外部集上具有稳定的 AUROC 优势。",
            f"- {first.title} 的外部特异度为 {first_test['specificity']:.4f}，"
            f"{second.title} 的外部敏感度为 {second_test['sensitivity']:.4f}；"
            "这反映了两者内部集阈值对应的工作点不同，不等同于排序性能有明显差异。",
            f"- 两模型外部预测一致率为 {external_agreement['prediction_agreement']:.1%}，"
            f"异常评分 Spearman 相关为 {external_agreement['probability_spearman']:.3f}，"
            "说明二者共享较多信息但仍存在互补分歧。",
            f"- 平均集成外部 AUROC={ensemble_test['auroc']:.4f}、AUPRC={ensemble_test['auprc']:.4f}。"
            f"相对 {first.title} 的 AUROC 差值95% CI为 "
            f"[{first_auc_delta['ci_low']:.4f}, {first_auc_delta['ci_high']:.4f}]，"
            f"相对 {second.title} 为 [{second_auc_delta['ci_low']:.4f}, {second_auc_delta['ci_high']:.4f}]；"
            f"AUPRC 差值区间分别为 [{first_ap_delta['ci_low']:.4f}, {first_ap_delta['ci_high']:.4f}] 和 "
            f"[{second_ap_delta['ci_low']:.4f}, {second_ap_delta['ci_high']:.4f}]。",
        ]
    )
    for model in models:
        fit_row = by_key[("fit", model.name)]
        test_row = by_key[("test", model.name)]
        lines.append(
            f"- {model.title} 从 fit 到外部集的 AUROC 变化为 "
            f"{test_row['auroc'] - fit_row['auroc']:+.4f}。这表示泛化差距，"
            "但在无法确认 fit 预测是否严格 out-of-fold 时，不直接解释为过拟合。"
        )
    generalization_gaps = {
        model.name: by_key[("fit", model.name)]["auroc"]
        - by_key[("test", model.name)]["auroc"]
        for model in models
    }
    larger_gap_model = max(models, key=lambda model: generalization_gaps[model.name])
    smaller_gap_model = min(models, key=lambda model: generalization_gaps[model.name])
    lines.append(
        f"- {larger_gap_model.title} 的 fit→外部 AUROC 降幅 "
        f"({generalization_gaps[larger_gap_model.name]:.4f}) 大于 "
        f"{smaller_gap_model.title} ({generalization_gaps[smaller_gap_model.name]:.4f})；"
        "这里仅将其描述为更明显的泛化差距。"
    )
    lines.extend(["", "## 健康距离、稳定性与校准", ""])
    distance_correlations: dict[str, float] = {}
    for model in models:
        data = model.splits["test"]
        healthy_median = float(np.median(data.health_distance[data.labels == 0]))
        abnormal_median = float(np.median(data.health_distance[data.labels == 1]))
        score_correlation = float(
            spearmanr(data.health_distance, data.probabilities).statistic
        )
        distance_correlations[model.name] = score_correlation
        result = stability["test"][model.name]
        lines.append(
            f"- {model.title} 外部集健康/异常人群 health distance 中位数分别为 "
            f"{healthy_median:.3f}/{abnormal_median:.3f}，与异常评分的 Spearman 相关为 "
            f"{score_correlation:.3f}。"
        )
        lines.append(
            f"- {model.title} 的 TTA instability 与错误指标 Spearman 相关为 "
            f"{result['error_spearman']:.3f}；仅保留 instability 最低50%患者时，"
            f"错误率为 {result['selective_error'][0]:.3f}，全覆盖错误率为 "
            f"{result['selective_error'][-1]:.3f}。"
        )
    stronger_distance_model = max(
        models, key=lambda model: abs(distance_correlations[model.name])
    )
    lines.extend(
        [
            f"- 两种骨干中，{stronger_distance_model.title} 的 health distance 与异常评分关系更强；"
            "该结果仍属于关联性描述。",
            "- 当前 instability 五分位错误率和 selective-risk 曲线均无稳定单调改善，"
            "因此不能把 TTA instability 单独作为可靠拒判依据。",
            f"- 外部 ECE 为 {first_test['ece']:.3f}（{first.title}）和 "
            f"{second_test['ece']:.3f}（{second.title}）。当前 `abnormal_probability` 更适合解释为"
            "异常评分，而不是已经临床校准的绝对概率。",
            "",
            "## 特征系数",
            "",
        ]
    )
    for model in models:
        order = np.argsort(np.abs(model.classifier_weights))[::-1][:5]
        features = "、".join(
            f"`{model.feature_names[index]}` ({model.classifier_weights[index]:+.3f})"
            for index in order
        )
        lines.append(f"- {model.title} 绝对权重最大的特征为：{features}。")
    lines.extend(
        [
            "- 系数是在标准化特征上的条件关联；由于各指标 logits、health distance 和"
            "outside fraction 相互相关，不应把单个系数解释为独立因果贡献。",
            "",
            "## 图表",
            "",
            "![Performance overview](figures/performance_overview.png)",
            "",
            "![ROC and PR curves](figures/roc_pr_curves.png)",
            "",
            "![Confusion matrices](figures/confusion_matrices.png)",
            "",
            "![Score distributions](figures/score_distributions.png)",
            "",
            "![Internal and external score distributions]"
            "(figures/score_distributions_internal_external.png)",
            "",
            "![Calibration](figures/calibration_curves.png)",
            "",
            "![Health distance](figures/health_distance.png)",
            "",
            "![TTA stability](figures/tta_stability.png)",
            "",
            "![Model agreement](figures/model_agreement.png)",
            "",
            "![Feature coefficients](figures/feature_coefficients.png)",
            "",
            "## 限制",
            "",
            "- 健康标签由11个指标全为0定义，不等价于完整临床健康状态。",
            "- 平均集成是在当前内部集选择阈值后的探索性结果，需要在新的独立队列继续验证。",
            "- 类别不平衡且异常患病率较高，AUPRC 应结合数据集阳性率解读。",
            "- 本分析用于研究与模型比较，不构成临床诊断结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if len(args.models) != 2:
        raise ValueError("--models currently requires exactly two model names")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    results_root = args.results_root.resolve(strict=True)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root / "diagnosis_analysis"
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = [load_model(results_root, name) for name in args.models]
    validate_alignment(models)

    labels = {split: models[0].splits[split].labels for split in SPLITS}
    ensemble_calibration = np.mean(
        [model.splits["calibration"].probabilities for model in models], axis=0
    )
    ensemble_threshold = select_youden_threshold(
        labels["calibration"], ensemble_calibration
    )
    model_order = [model.name for model in models] + ["ensemble"]
    thresholds = {model.name: model.splits["test"].threshold for model in models}
    thresholds["ensemble"] = float(ensemble_threshold)
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        probabilities[split] = {
            model.name: model.splits[split].probabilities for model in models
        }
        probabilities[split]["ensemble"] = np.mean(
            [model.splits[split].probabilities for model in models], axis=0
        )

    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        intervals, comparisons = bootstrap_split(
            labels[split],
            probabilities[split],
            samples=args.bootstrap_samples,
            rng=rng,
        )
        for model in model_order:
            values = metric_values(
                labels[split], probabilities[split][model], thresholds[model]
            )
            values.update(
                {
                    "split": split,
                    "model": model,
                    "auroc_ci_low": intervals[(model, "auroc")][0],
                    "auroc_ci_high": intervals[(model, "auroc")][1],
                    "auprc_ci_low": intervals[(model, "auprc")][0],
                    "auprc_ci_high": intervals[(model, "auprc")][1],
                }
            )
            ordered = {"split": split, "model": model}
            ordered.update({key: value for key, value in values.items() if key not in ordered})
            metric_rows.append(ordered)
        for row in comparisons:
            comparison_rows.append({"split": split, **row})

    stability = {
        split: {
            model.name: instability_analysis(model.splits[split]) for model in models
        }
        for split in SPLITS
    }
    agreement: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        first, second = (model.splits[split] for model in models)
        agreement[split] = {
            "prediction_agreement": float(np.mean(first.predictions == second.predictions)),
            "prediction_disagreement": float(np.mean(first.predictions != second.predictions)),
            "probability_spearman": float(
                spearmanr(first.probabilities, second.probabilities).statistic
            ),
        }

    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "bootstrap_comparisons.csv", comparison_rows)
    summary = {
        "models": [model.name for model in models],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "ensemble_threshold": float(ensemble_threshold),
        "metrics": metric_rows,
        "bootstrap_comparisons": comparison_rows,
        "agreement": agreement,
        "stability": stability,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_performance(
        metric_rows, model_order, figure_dir / "performance_overview.png", args.dpi
    )
    plot_roc_pr(
        labels,
        probabilities,
        model_order,
        figure_dir / "roc_pr_curves.png",
        args.dpi,
    )
    plot_confusions(
        labels,
        probabilities,
        thresholds,
        model_order,
        figure_dir / "confusion_matrices.png",
        args.dpi,
    )
    plot_probability_distributions(
        labels,
        probabilities,
        model_order,
        figure_dir / "score_distributions.png",
        args.dpi,
    )
    plot_probability_distributions(
        labels,
        probabilities,
        [model.name for model in models],
        figure_dir / "score_distributions_internal_external.png",
        args.dpi,
        splits=("calibration", "test"),
        title="Internal and external abnormal-score distributions",
        split_titles={
            "calibration": "Internal validation",
            "test": "External validation",
        },
    )
    plot_calibration(
        labels,
        probabilities,
        model_order,
        figure_dir / "calibration_curves.png",
        args.dpi,
    )
    plot_health_distance(models, figure_dir / "health_distance.png", args.dpi)
    plot_stability(models, stability, figure_dir / "tta_stability.png", args.dpi)
    plot_agreement(models, figure_dir / "model_agreement.png", args.dpi)
    plot_coefficients(models, figure_dir / "feature_coefficients.png", args.dpi)
    write_report(
        output_dir / "analysis_report.md",
        models=models,
        metric_rows=metric_rows,
        comparison_rows=comparison_rows,
        stability=stability,
        agreement=agreement,
    )
    print(
        f"saved diagnosis analysis for {', '.join(model.title for model in models)} "
        f"to {output_dir}",
        flush=True,
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
