#!/usr/bin/env python3
"""Plot a shared 2-D t-SNE of 16-D fit mean-logit vectors across models."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


CLASS_STYLES = {
    0: ("Healthy", "#4C78A8"),
    1: ("Abnormal", "#E45756"),
}


@dataclass(frozen=True)
class FitPredictions:
    name: str
    path: Path
    features: np.ndarray
    labels: np.ndarray
    source_rows: np.ndarray
    indicators: tuple[str, ...]
    seeds: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-root",
        type=Path,
        required=True,
        help="Root containing <model>/diagnosis/fit_predictions.pt",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model directory names (default: automatic discovery)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Default: <fit-root parent>/fit_16d_tsne.jpg",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive")
    if args.max_iter < 250:
        parser.error("--max-iter must be at least 250")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _tensor(payload: dict[str, Any], key: str, path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: missing tensor {key!r}")
    return value


def discover_models(root: Path, requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested is not None:
        names = tuple(requested)
    else:
        names = tuple(
            sorted(
                child.name
                for child in root.iterdir()
                if (child / "diagnosis" / "fit_predictions.pt").is_file()
            )
        )
    if not names:
        raise ValueError(f"no model fit predictions found under {root}")
    if len(names) != len(set(names)):
        raise ValueError("--models must not contain duplicates")
    return names


def load_fit_predictions(root: Path, name: str) -> FitPredictions:
    path = (root / name / "diagnosis" / "fit_predictions.pt").resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"fit predictions must contain a dictionary: {path}")
    features = _tensor(payload, "mean_logits", path)
    labels = _tensor(payload, "labels", path)
    source_rows = _tensor(payload, "source_row", path)
    seeds = _tensor(payload, "seeds", path)
    indicators = tuple(payload.get("indicators", ()))
    if features.ndim != 2 or features.shape[1] != 16:
        raise ValueError(f"{path}: mean_logits must have shape [patients, 16]")
    if labels.shape != (features.shape[0],) or source_rows.shape != labels.shape:
        raise ValueError(f"{path}: patient tensors have inconsistent shapes")
    if len(indicators) != features.shape[1]:
        raise ValueError(f"{path}: indicators do not match mean_logits")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{path}: mean_logits contains non-finite values")
    if not bool(torch.isin(labels, torch.tensor([0, 1])).all()):
        raise ValueError(f"{path}: fit labels must all be healthy/abnormal")
    return FitPredictions(
        name=name,
        path=path,
        features=features.float().numpy(),
        labels=labels.numpy(),
        source_rows=source_rows.numpy(),
        indicators=indicators,
        seeds=seeds.numpy(),
    )


def validate_alignment(datasets: Sequence[FitPredictions]) -> None:
    reference = datasets[0]
    for current in datasets[1:]:
        if current.indicators != reference.indicators:
            raise ValueError(f"indicator order differs in {current.path}")
        if not np.array_equal(current.source_rows, reference.source_rows):
            raise ValueError(f"patient source_rows differ in {current.path}")
        if not np.array_equal(current.labels, reference.labels):
            raise ValueError(f"health labels differ in {current.path}")
        if not np.array_equal(current.seeds, reference.seeds):
            raise ValueError(f"inference seeds differ in {current.path}")


def standardize_per_model(
    datasets: Sequence[FitPredictions],
) -> tuple[list[np.ndarray], dict[str, dict[str, list[float]]]]:
    standardized: list[np.ndarray] = []
    metadata: dict[str, dict[str, list[float]]] = {}
    for data in datasets:
        mean = data.features.mean(axis=0, dtype=np.float64)
        std = data.features.std(axis=0, dtype=np.float64)
        std = np.where(std > 1e-8, std, 1.0)
        standardized.append(
            ((data.features - mean) / std).astype(np.float32, copy=False)
        )
        metadata[data.name] = {
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
        }
    return standardized, metadata


def fit_joint_tsne(
    features: Sequence[np.ndarray],
    *,
    perplexity: float,
    max_iter: int,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    combined = np.concatenate(features, axis=0)
    effective_perplexity = min(float(perplexity), float(combined.shape[0] - 1))
    arguments: dict[str, Any] = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "method": "barnes_hut",
        "verbose": 1,
    }
    signature = inspect.signature(TSNE).parameters
    arguments["max_iter" if "max_iter" in signature else "n_iter"] = max_iter
    if "n_jobs" in signature:
        arguments["n_jobs"] = -1
    reducer = TSNE(**arguments)
    coordinates = reducer.fit_transform(combined).astype(np.float32, copy=False)
    split_coordinates: list[np.ndarray] = []
    offset = 0
    for values in features:
        split_coordinates.append(coordinates[offset : offset + len(values)])
        offset += len(values)
    return split_coordinates, {
        "requested_perplexity": float(perplexity),
        "effective_perplexity": effective_perplexity,
        "max_iter": int(max_iter),
        "iterations_completed": int(reducer.n_iter_),
        "kl_divergence": float(reducer.kl_divergence_),
        "random_seed": int(seed),
    }


def plot_comparison(
    datasets: Sequence[FitPredictions],
    coordinates: Sequence[np.ndarray],
    output: Path,
    dpi: int,
) -> None:
    combined = np.concatenate(coordinates, axis=0)
    limits = []
    for index in range(2):
        low, high = np.quantile(combined[:, index], (0.002, 0.998))
        padding = 0.04 * float(high - low)
        limits.append((float(low - padding), float(high + padding)))
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7.0 * len(datasets), 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, data, embedding in zip(axes, datasets, coordinates):
        for label in (1, 0):
            mask = data.labels == label
            class_name, color = CLASS_STYLES[label]
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=3.0,
                alpha=0.30 if label == 1 else 0.42,
                color=color,
                edgecolors="none",
                rasterized=True,
                label=f"{class_name} (n={int(mask.sum()):,})",
            )
        axis.set_title(data.name.replace("_dali_balanced", "").replace("_", " "))
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, markerscale=3, loc="best")
    figure.suptitle(
        "Fit set: joint t-SNE of 16-D mean logits across 50 inference seeds",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95},
    )
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    root = args.fit_root.expanduser().resolve(strict=True)
    model_names = discover_models(root, args.models)
    datasets = [load_fit_predictions(root, name) for name in model_names]
    validate_alignment(datasets)
    standardized, standardization = standardize_per_model(datasets)
    coordinates, tsne_metadata = fit_joint_tsne(
        standardized,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    output = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else root.parent / "fit_16d_tsne.jpg"
    )
    plot_comparison(datasets, coordinates, output, args.dpi)
    labels = datasets[0].labels
    summary = {
        "figure": str(output),
        "fit_root": str(root),
        "models": list(model_names),
        "feature": "mean_logits across 50 inference seeds",
        "feature_dimensions": 16,
        "indicators": list(datasets[0].indicators),
        "patients": int(len(labels)),
        "healthy": int((labels == 0).sum()),
        "abnormal": int((labels == 1).sum()),
        "embedding_scope": "joint across models after per-model standardization",
        "standardization": standardization,
        "tsne": tsne_metadata,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved fit t-SNE figure to {output}", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
