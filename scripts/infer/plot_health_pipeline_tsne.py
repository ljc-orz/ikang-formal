#!/usr/bin/env python3
"""Plot a joint t-SNE embedding from health-pipeline aggregate files."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.infer.plot_health_pipeline_pca import (  # noqa: E402
    CLASS_STYLES,
    DATASETS,
    load_health_labels,
)


# Fill this path to run the script without --health-pipeline-output.
HEALTH_PIPELINE_OUTPUT: Path | None = None


@dataclass(frozen=True)
class TsneDataset:
    name: str
    title: str
    path: Path
    features: np.ndarray
    coordinates: np.ndarray | None
    source_rows: np.ndarray
    targets: tuple[str, ...]
    indicators: tuple[str, ...]
    seeds: np.ndarray
    eye: str
    transform: str
    source_parquet: str | None
    pca_components: np.ndarray | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--health-pipeline-output",
        type=Path,
        default=HEALTH_PIPELINE_OUTPUT,
        help="Directory containing aggregated/ and diagnosis/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <health-pipeline-output>/tsne_figures",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Override source_parquet metadata for all three datasets",
    )
    parser.add_argument(
        "--feature-source",
        choices=("raw", "pca"),
        default="raw",
        help="raw: mean original indicator logits; pca: mean X_prime (default: raw)",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.health_pipeline_output is None:
        parser.error(
            "set HEALTH_PIPELINE_OUTPUT at the top of the script or provide "
            "--health-pipeline-output"
        )
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


def load_tsne_dataset(
    path: Path,
    name: str,
    title: str,
    feature_source: str,
) -> TsneDataset:
    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"aggregate file must contain a dictionary: {resolved}")
    feature_key = "X" if feature_source == "raw" else "X_prime"
    values = _tensor(payload, feature_key, resolved)
    source_rows = _tensor(payload, "source_row", resolved)
    seeds = _tensor(payload, "seeds", resolved)
    if values.ndim != 3:
        raise ValueError(
            f"{resolved}: {feature_key} must have shape [seeds, patients, features]"
        )
    if values.shape[2] < 1:
        raise ValueError(f"{resolved}: {feature_key} has no feature columns")
    if source_rows.dtype != torch.int64 or source_rows.shape != (values.shape[1],):
        raise ValueError(f"{resolved}: invalid source_row")
    if source_rows.numel() != torch.unique(source_rows).numel():
        raise ValueError(f"{resolved}: duplicate source_row values")
    if seeds.dtype != torch.int64 or seeds.shape != (values.shape[0],):
        raise ValueError(f"{resolved}: seeds do not match {feature_key}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{resolved}: {feature_key} contains non-finite values")
    targets = tuple(payload.get("targets", ()))
    indicators = tuple(payload.get("indicators", ()))
    if not targets or len(targets) != len(indicators):
        raise ValueError(f"{resolved}: invalid target/indicator metadata")
    if feature_source == "raw" and values.shape[2] != len(indicators):
        raise ValueError(
            f"{resolved}: raw feature count does not match indicator metadata"
        )
    pca_components = None
    if feature_source == "pca":
        reducer = payload.get("reducer")
        if not isinstance(reducer, dict) or reducer.get("method") != "pca":
            raise ValueError(f"{resolved}: missing PCA reducer metadata")
        components = reducer.get("components")
        if (
            not isinstance(components, torch.Tensor)
            or components.ndim != 2
            or components.shape[0] != values.shape[2]
        ):
            raise ValueError(f"{resolved}: PCA components do not match X_prime")
        pca_components = components.to(dtype=torch.float32).numpy()
    stored_split = str(payload.get("split", ""))
    if stored_split != name:
        raise ValueError(
            f"{resolved}: split metadata is {stored_split!r}, expected {name!r}"
        )
    source_parquet = payload.get("source_parquet")
    return TsneDataset(
        name=name,
        title=title,
        path=resolved,
        features=values.to(dtype=torch.float32).mean(dim=0).numpy(),
        coordinates=None,
        source_rows=source_rows.numpy(),
        targets=targets,
        indicators=indicators,
        seeds=seeds.numpy(),
        eye=str(payload.get("eye", "")),
        transform=str(payload.get("transform", "")),
        source_parquet=None if source_parquet is None else str(source_parquet),
        pca_components=pca_components,
    )


def validate_shared_features(datasets: Sequence[TsneDataset]) -> None:
    reference = datasets[0]
    for current in datasets[1:]:
        for field in ("targets", "indicators", "eye", "transform"):
            if getattr(reference, field) != getattr(current, field):
                raise ValueError(
                    f"{field} differs between {reference.path} and {current.path}"
                )
        if not np.array_equal(reference.seeds, current.seeds):
            raise ValueError(f"seeds differ between {reference.path} and {current.path}")
        if reference.features.shape[1] != current.features.shape[1]:
            raise ValueError(
                f"feature dimensions differ between {reference.path} and {current.path}"
            )
        if reference.pca_components is not None and not np.array_equal(
            reference.pca_components, current.pca_components
        ):
            raise ValueError(
                f"PCA components differ between {reference.path} and {current.path}"
            )


def standardize_from_train(
    datasets: Sequence[TsneDataset],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    feature_mean = datasets[0].features.mean(axis=0, dtype=np.float64)
    feature_std = datasets[0].features.std(axis=0, dtype=np.float64)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    standardized = [
        ((data.features - feature_mean) / feature_std).astype(np.float32, copy=False)
        for data in datasets
    ]
    return standardized, feature_mean, feature_std


def fit_joint_tsne(
    datasets: Sequence[TsneDataset],
    *,
    perplexity: float,
    max_iter: int,
    seed: int,
) -> tuple[list[TsneDataset], dict[str, Any]]:
    standardized, feature_mean, feature_std = standardize_from_train(datasets)
    combined = np.concatenate(standardized, axis=0)
    if combined.shape[0] < 3:
        raise ValueError("joint t-SNE requires at least three patients")
    effective_perplexity = min(float(perplexity), float(combined.shape[0] - 1))
    arguments: dict[str, Any] = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "method": "barnes_hut",
    }
    # scikit-learn 1.5 renamed n_iter to max_iter; the supported project range
    # includes both APIs.
    if "max_iter" in inspect.signature(TSNE).parameters:
        arguments["max_iter"] = max_iter
    else:
        arguments["n_iter"] = max_iter
    reducer = TSNE(**arguments)
    coordinates = reducer.fit_transform(combined).astype(np.float32, copy=False)

    output: list[TsneDataset] = []
    offset = 0
    for data in datasets:
        count = data.features.shape[0]
        output.append(replace(data, coordinates=coordinates[offset : offset + count]))
        offset += count
    metadata = {
        "fit_scope": "joint train + internal_validation + external_validation",
        "standardization": "training-set feature mean and standard deviation",
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "requested_perplexity": float(perplexity),
        "effective_perplexity": effective_perplexity,
        "max_iter": int(max_iter),
        "iterations_completed": int(reducer.n_iter_),
        "kl_divergence": float(reducer.kl_divergence_),
        "random_seed": int(seed),
        "init": "pca",
        "learning_rate": "auto",
        "method": "barnes_hut",
    }
    return output, metadata


def shared_axis_limits(datasets: Sequence[TsneDataset]) -> tuple[tuple[float, float], ...]:
    coordinates = np.concatenate(
        [data.coordinates for data in datasets if data.coordinates is not None], axis=0
    )
    limits = []
    for index in range(2):
        low, high = np.quantile(coordinates[:, index], (0.005, 0.995))
        span = float(high - low)
        padding = 0.05 * span if span > 0 else 1.0
        limits.append((float(low - padding), float(high + padding)))
    return tuple(limits)


def plot_dataset(
    data: TsneDataset,
    labels: np.ndarray,
    limits: Sequence[tuple[float, float]],
    output: Path,
    dpi: int,
) -> None:
    if data.coordinates is None:
        raise RuntimeError(f"t-SNE coordinates were not fitted for {data.name}")
    figure, axis = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    for label in (-1, 0, 1):
        mask = labels == label
        if not mask.any():
            continue
        class_name, color = CLASS_STYLES[label]
        axis.scatter(
            data.coordinates[mask, 0],
            data.coordinates[mask, 1],
            s=7,
            alpha=0.28 if label >= 0 else 0.18,
            color=color,
            edgecolors="none",
            rasterized=True,
            label=f"{class_name} (n={mask.sum():,})",
        )
        if label >= 0:
            center = np.median(data.coordinates[mask], axis=0)
            axis.scatter(
                center[0],
                center[1],
                marker="X",
                s=90,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.set_xlim(limits[0])
    axis.set_ylim(limits[1])
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, markerscale=2, fontsize=9)
    axis.set_title(data.title)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    pipeline_output = args.health_pipeline_output.resolve(strict=True)
    aggregate_dir = pipeline_output / "aggregated"
    datasets = [
        load_tsne_dataset(
            aggregate_dir / f"{name}.pt", name, title, args.feature_source
        )
        for name, title in DATASETS
    ]
    validate_shared_features(datasets)
    labels = {
        data.name: load_health_labels(data, args.parquet) for data in datasets
    }
    datasets, tsne_metadata = fit_joint_tsne(
        datasets,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    limits = shared_axis_limits(datasets)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else pipeline_output / "tsne_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for data in datasets:
        plot_dataset(
            data,
            labels[data.name],
            limits,
            output_dir / f"tsne_{data.name}.png",
            args.dpi,
        )
    summary = {
        "coordinate_summary": (
            "joint t-SNE of each patient's mean feature vector across seeds"
        ),
        "feature_source": args.feature_source,
        "seed_count": len(datasets[0].seeds),
        "seeds": datasets[0].seeds.tolist(),
        "indicators": list(datasets[0].indicators),
        "shared_axis_quantiles": [0.005, 0.995],
        "tsne": tsne_metadata,
        "datasets": {
            data.name: {
                "patients": int(len(labels[data.name])),
                "healthy": int((labels[data.name] == 0).sum()),
                "abnormal": int((labels[data.name] == 1).sum()),
                "incomplete_label": int((labels[data.name] == -1).sum()),
                "figure": f"tsne_{data.name}.png",
            }
            for data in datasets
        },
    }
    (output_dir / "tsne_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved joint train/internal/external t-SNE figures and summary to {output_dir}",
        flush=True,
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
