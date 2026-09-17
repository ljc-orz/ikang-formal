#!/usr/bin/env python3
"""Plot patient-level PCA projections from a health-pipeline output directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


# Fill this path to run the script without --health-pipeline-output.
HEALTH_PIPELINE_OUTPUT: Path | None = None


DATASETS = (
    ("train", "Train"),
    ("internal_validation", "Internal validation"),
    ("external_validation", "External validation"),
)
CLASS_STYLES = {
    -1: ("Incomplete label", "#9D9D9D"),
    0: ("Healthy", "#4C78A8"),
    1: ("Abnormal", "#E45756"),
}


@dataclass(frozen=True)
class PcaDataset:
    name: str
    title: str
    path: Path
    coordinates: np.ndarray
    source_rows: np.ndarray
    targets: tuple[str, ...]
    indicators: tuple[str, ...]
    seeds: np.ndarray
    eye: str
    transform: str
    source_parquet: str | None
    components: np.ndarray
    explained_variance_ratio: np.ndarray


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
        help="Default: <health-pipeline-output>/pca_figures",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Override source_parquet metadata for all three datasets",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180)",
    )
    args = parser.parse_args(argv)
    if args.health_pipeline_output is None:
        parser.error(
            "set HEALTH_PIPELINE_OUTPUT at the top of the script or provide "
            "--health-pipeline-output"
        )
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _tensor(payload: dict[str, Any], key: str, path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: missing tensor {key!r}")
    return value


def load_pca_dataset(path: Path, name: str, title: str) -> PcaDataset:
    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"aggregate file must contain a dictionary: {resolved}")
    projected = _tensor(payload, "X_prime", resolved)
    source_rows = _tensor(payload, "source_row", resolved)
    seeds = _tensor(payload, "seeds", resolved)
    if projected.ndim != 3:
        raise ValueError(f"{resolved}: X_prime must have shape [seeds, patients, PCs]")
    if projected.shape[2] < 2:
        raise ValueError(f"{resolved}: at least two PCA components are required")
    if source_rows.dtype != torch.int64 or source_rows.shape != (projected.shape[1],):
        raise ValueError(f"{resolved}: invalid source_row")
    if source_rows.numel() != torch.unique(source_rows).numel():
        raise ValueError(f"{resolved}: duplicate source_row values")
    if seeds.dtype != torch.int64 or seeds.shape != (projected.shape[0],):
        raise ValueError(f"{resolved}: seeds do not match X_prime")
    if not bool(torch.isfinite(projected).all()):
        raise ValueError(f"{resolved}: X_prime contains non-finite values")
    targets = tuple(payload.get("targets", ()))
    indicators = tuple(payload.get("indicators", ()))
    if not targets or len(targets) != len(indicators):
        raise ValueError(f"{resolved}: invalid target/indicator metadata")
    stored_split = str(payload.get("split", ""))
    if stored_split != name:
        raise ValueError(
            f"{resolved}: split metadata is {stored_split!r}, expected {name!r}"
        )
    reducer = payload.get("reducer")
    if not isinstance(reducer, dict) or reducer.get("method") != "pca":
        raise ValueError(f"{resolved}: missing PCA reducer metadata")
    components = reducer.get("components")
    explained_ratio = reducer.get("explained_variance_ratio")
    if not isinstance(components, torch.Tensor) or components.ndim != 2:
        raise ValueError(f"{resolved}: PCA components must be a shared 2-D tensor")
    if not isinstance(explained_ratio, torch.Tensor) or explained_ratio.ndim != 1:
        raise ValueError(f"{resolved}: invalid explained_variance_ratio")
    component_count = projected.shape[2]
    if components.shape[0] != component_count or explained_ratio.shape[0] != component_count:
        raise ValueError(f"{resolved}: reducer dimensions do not match X_prime")
    source_parquet = payload.get("source_parquet")
    return PcaDataset(
        name=name,
        title=title,
        path=resolved,
        coordinates=projected.to(dtype=torch.float32).mean(dim=0).numpy(),
        source_rows=source_rows.numpy(),
        targets=targets,
        indicators=indicators,
        seeds=seeds.numpy(),
        eye=str(payload.get("eye", "")),
        transform=str(payload.get("transform", "")),
        source_parquet=None if source_parquet is None else str(source_parquet),
        components=components.to(dtype=torch.float32).numpy(),
        explained_variance_ratio=explained_ratio.to(dtype=torch.float32).numpy(),
    )


def validate_shared_pca(datasets: Sequence[PcaDataset]) -> None:
    reference = datasets[0]
    for current in datasets[1:]:
        for field in ("targets", "indicators", "eye", "transform"):
            if getattr(reference, field) != getattr(current, field):
                raise ValueError(
                    f"{field} differs between {reference.path} and {current.path}"
                )
        if not np.array_equal(reference.seeds, current.seeds):
            raise ValueError(f"seeds differ between {reference.path} and {current.path}")
        if not np.array_equal(reference.components, current.components):
            raise ValueError(
                f"PCA components differ between {reference.path} and {current.path}; "
                "the datasets are not in the same coordinate system"
            )
        if not np.array_equal(
            reference.explained_variance_ratio, current.explained_variance_ratio
        ):
            raise ValueError(
                f"explained variance differs between {reference.path} and {current.path}"
            )


def load_health_labels(data: PcaDataset, parquet_override: Path | None) -> np.ndarray:
    if parquet_override is None:
        if data.source_parquet is None:
            raise ValueError(f"{data.path} has no source_parquet; provide --parquet")
        parquet = Path(data.source_parquet)
    else:
        parquet = parquet_override
    resolved = parquet.expanduser().resolve(strict=True)
    frame = pd.read_parquet(resolved, columns=list(data.targets))
    rows = data.source_rows
    if rows.size and (rows.min() < 0 or rows.max() >= len(frame)):
        raise ValueError(f"source_row is outside parquet bounds in {data.path}")
    values = frame.iloc[rows].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    valid = np.logical_or(values == 0, values == 1).all(axis=1)
    labels = np.full(len(rows), -1, dtype=np.int64)
    labels[valid] = (values[valid] == 1).any(axis=1).astype(np.int64)
    return labels


def shared_axis_limits(datasets: Sequence[PcaDataset]) -> list[tuple[float, float]]:
    all_coordinates = np.concatenate([data.coordinates for data in datasets], axis=0)
    limits: list[tuple[float, float]] = []
    for index in range(all_coordinates.shape[1]):
        low, high = np.quantile(all_coordinates[:, index], (0.005, 0.995))
        span = float(high - low)
        padding = 0.05 * span if span > 0 else 1.0
        limits.append((float(low - padding), float(high + padding)))
    return limits


def component_label(index: int, explained_ratio: np.ndarray) -> str:
    return f"PC{index + 1} ({explained_ratio[index]:.1%})"


def plot_dataset(
    data: PcaDataset,
    labels: np.ndarray,
    limits: Sequence[tuple[float, float]],
    output: Path,
    dpi: int,
) -> None:
    component_count = data.coordinates.shape[1]
    pairs = [(0, 1)] if component_count == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5.5))
    axes = np.atleast_1d(axes)
    for axis, (x_index, y_index) in zip(axes, pairs):
        for label in (-1, 0, 1):
            mask = labels == label
            if not mask.any():
                continue
            class_name, color = CLASS_STYLES[label]
            axis.scatter(
                data.coordinates[mask, x_index],
                data.coordinates[mask, y_index],
                s=7,
                alpha=0.28 if label >= 0 else 0.18,
                color=color,
                edgecolors="none",
                rasterized=True,
                label=f"{class_name} (n={mask.sum():,})",
            )
            if label >= 0:
                center = np.median(data.coordinates[mask][:, (x_index, y_index)], axis=0)
                axis.scatter(
                    center[0],
                    center[1],
                    marker="X",
                    s=90,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
        axis.set_xlabel(component_label(x_index, data.explained_variance_ratio))
        axis.set_ylabel(component_label(y_index, data.explained_variance_ratio))
        axis.set_xlim(limits[x_index])
        axis.set_ylim(limits[y_index])
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, markerscale=2, fontsize=9)
    fig.suptitle(
        f"{data.title}: mean PCA coordinates across {len(data.seeds)} seeds",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    pipeline_output = args.health_pipeline_output.resolve(strict=True)
    aggregate_dir = pipeline_output / "aggregated"
    datasets = [
        load_pca_dataset(aggregate_dir / f"{name}.pt", name, title)
        for name, title in DATASETS
    ]
    validate_shared_pca(datasets)
    labels = {
        data.name: load_health_labels(data, args.parquet) for data in datasets
    }
    limits = shared_axis_limits(datasets)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else pipeline_output / "pca_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for data in datasets:
        plot_dataset(
            data,
            labels[data.name],
            limits,
            output_dir / f"pca_{data.name}.png",
            args.dpi,
        )
    summary = {
        "coordinate_summary": "mean X_prime across seeds for each patient",
        "shared_axis_quantiles": [0.005, 0.995],
        "seed_count": len(datasets[0].seeds),
        "seeds": datasets[0].seeds.tolist(),
        "indicators": list(datasets[0].indicators),
        "explained_variance_ratio": datasets[0].explained_variance_ratio.tolist(),
        "datasets": {
            data.name: {
                "patients": int(len(labels[data.name])),
                "healthy": int((labels[data.name] == 0).sum()),
                "abnormal": int((labels[data.name] == 1).sum()),
                "incomplete_label": int((labels[data.name] == -1).sum()),
                "figure": f"pca_{data.name}.png",
            }
            for data in datasets
        },
    }
    (output_dir / "pca_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved train/internal/external PCA figures and summary to {output_dir}",
        flush=True,
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
