#!/usr/bin/env python3
"""Combine per-target inference logits and reduce them to patient embeddings."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA


PREDICTION_NAME = re.compile(
    r"^.+\.(train|internal_validation|external_validation)\.seed-(\d+)\.pt$"
)


@dataclass(frozen=True)
class InputLayout:
    indicators: tuple[str, ...]
    split: str
    seeds: tuple[int, ...]
    files: dict[str, dict[int, Path]]


@dataclass(frozen=True)
class LoadedPredictions:
    logits: torch.Tensor
    source_rows: torch.Tensor
    targets: tuple[str, ...]
    transform: str
    source_parquet: str | None
    checkpoints: dict[str, str]


@dataclass(frozen=True)
class ReducedPredictions:
    values: torch.Tensor
    components: torch.Tensor
    feature_mean: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per indicator (alt, bmi, ...)",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--indicators",
        nargs="+",
        help="Indicator subdirectories and feature order (default: sorted discovery)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        help=(
            "Seeds to include, or 'seq FIRST LAST' with inclusive endpoints "
            "(default: require and use every common seed)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "internal_validation", "external_validation"),
        help="Required when the input contains more than one split",
    )
    parser.add_argument(
        "--eye",
        choices=("left", "right", "mean"),
        default="mean",
        help="Logit used for each indicator (default: mean of left and right)",
    )
    parser.add_argument(
        "--method",
        choices=("pca",),
        default="pca",
        help="Dimensionality-reduction method (default: pca)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Output dimensions M (default: 3)",
    )
    parser.add_argument(
        "--pca-fit",
        choices=("joint", "mean", "per-seed"),
        default="joint",
        help=(
            "PCA fitting data: all seed/patient rows, the seed mean, or one PCA "
            "per seed (default: joint)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.seeds = parse_seed_selection(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_seed_selection(values: Sequence[str] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    if values[0] == "seq":
        if len(values) != 3:
            raise ValueError("--seeds seq requires exactly FIRST and LAST")
        try:
            first, last = int(values[1]), int(values[2])
        except ValueError as exc:
            raise ValueError("--seeds seq FIRST LAST must use integers") from exc
        if first > last:
            raise ValueError("--seeds seq requires FIRST <= LAST")
        seeds = tuple(range(first, last + 1))
    else:
        try:
            seeds = tuple(int(value) for value in values)
        except ValueError as exc:
            raise ValueError("--seeds values must be integers") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be non-negative")
    return seeds


def _prediction_identity(path: Path) -> tuple[str, int] | None:
    match = PREDICTION_NAME.match(path.name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def discover_layout(
    input_dir: Path,
    *,
    indicators: Sequence[str] | None,
    split: str | None,
    seeds: Sequence[int] | None,
) -> InputLayout:
    root = input_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"--input-dir is not a directory: {root}")

    if indicators is None:
        indicator_names = tuple(
            sorted(
                child.name
                for child in root.iterdir()
                if child.is_dir()
                and any(_prediction_identity(path) for path in child.glob("*.pt"))
            )
        )
    else:
        indicator_names = tuple(indicators)
    if not indicator_names:
        raise ValueError(f"no indicator directories found in {root}")
    if len(indicator_names) != len(set(indicator_names)):
        raise ValueError("--indicators must not contain duplicates")

    records: dict[str, dict[str, dict[int, Path]]] = {}
    observed_splits: set[str] = set()
    for indicator in indicator_names:
        directory = root / indicator
        if not directory.is_dir():
            raise ValueError(f"indicator directory does not exist: {directory}")
        by_split: dict[str, dict[int, Path]] = {}
        for path in sorted(directory.glob("*.pt")):
            identity = _prediction_identity(path)
            if identity is None:
                continue
            file_split, seed = identity
            seed_files = by_split.setdefault(file_split, {})
            if seed in seed_files:
                raise ValueError(
                    f"multiple {file_split} files for seed {seed} in {directory}"
                )
            seed_files[seed] = path.resolve()
            observed_splits.add(file_split)
        if not by_split:
            raise ValueError(f"no inference .pt files found in {directory}")
        records[indicator] = by_split

    if split is None:
        if len(observed_splits) != 1:
            choices = ", ".join(sorted(observed_splits))
            raise ValueError(
                f"input contains multiple splits ({choices}); specify --split"
            )
        selected_split = next(iter(observed_splits))
    else:
        selected_split = split

    files: dict[str, dict[int, Path]] = {}
    for indicator in indicator_names:
        if selected_split not in records[indicator]:
            raise ValueError(
                f"indicator {indicator!r} has no {selected_split!r} predictions"
            )
        files[indicator] = records[indicator][selected_split]

    if seeds is None:
        selected_seeds = tuple(sorted(files[indicator_names[0]]))
        reference_set = set(selected_seeds)
        for indicator in indicator_names[1:]:
            current = set(files[indicator])
            if current != reference_set:
                missing = sorted(reference_set - current)
                extra = sorted(current - reference_set)
                raise ValueError(
                    f"seed set differs for {indicator!r}: missing={missing}, extra={extra}"
                )
    else:
        selected_seeds = tuple(seeds)
        if len(selected_seeds) != len(set(selected_seeds)):
            raise ValueError("--seeds must not contain duplicates")
        if any(seed < 0 for seed in selected_seeds):
            raise ValueError("--seeds must be non-negative")
        for indicator in indicator_names:
            missing = [seed for seed in selected_seeds if seed not in files[indicator]]
            if missing:
                raise ValueError(
                    f"indicator {indicator!r} is missing requested seeds: {missing}"
                )
    if not selected_seeds:
        raise ValueError("no inference seeds selected")

    return InputLayout(
        indicators=indicator_names,
        split=selected_split,
        seeds=selected_seeds,
        files=files,
    )


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"prediction file does not contain a dictionary: {path}")
    return payload


def _select_eye(logits: torch.Tensor, eye_order: Sequence[str], eye: str) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"logits must have shape [patients, 2], got {tuple(logits.shape)}")
    order = tuple(eye_order)
    if len(order) != 2 or set(order) != {"left", "right"}:
        raise ValueError(f"invalid eye_order: {order!r}")
    logits = logits.to(dtype=torch.float32)
    if eye == "mean":
        return logits.mean(dim=1)
    return logits[:, order.index(eye)]


def load_prediction_cube(layout: InputLayout, eye: str) -> LoadedPredictions:
    canonical_rows: torch.Tensor | None = None
    seed_matrices: list[torch.Tensor] = []
    target_by_indicator: dict[str, str] = {}
    checkpoint_by_indicator: dict[str, str] = {}
    reference_transform: str | None = None
    reference_parquet: str | None = None
    parquet_initialized = False

    for seed in layout.seeds:
        columns: list[torch.Tensor] = []
        for indicator in layout.indicators:
            path = layout.files[indicator][seed]
            payload = _load_payload(path)
            file_seed = int(payload.get("seed", -1))
            file_split = str(payload.get("split", ""))
            if file_seed != seed or file_split != layout.split:
                raise ValueError(
                    f"filename/metadata mismatch in {path}: "
                    f"seed={file_seed}, split={file_split!r}"
                )

            source_rows = payload.get("source_row")
            logits = payload.get("logits")
            if not isinstance(source_rows, torch.Tensor) or source_rows.ndim != 1:
                raise ValueError(f"invalid source_row tensor in {path}")
            if source_rows.dtype != torch.int64:
                raise ValueError(f"source_row must be int64 in {path}")
            if source_rows.numel() != torch.unique(source_rows).numel():
                raise ValueError(f"duplicate source_row values in {path}")
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"invalid logits tensor in {path}")
            if logits.shape[0] != source_rows.numel():
                raise ValueError(f"source_row/logits length mismatch in {path}")

            row_order = torch.argsort(source_rows)
            sorted_rows = source_rows[row_order]
            if canonical_rows is None:
                canonical_rows = sorted_rows
            elif not torch.equal(canonical_rows, sorted_rows):
                raise ValueError(f"patient source_row set differs in {path}")

            values = _select_eye(logits, payload.get("eye_order", ()), eye)[row_order]
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"non-finite logits found in {path}")
            columns.append(values)

            target = str(payload.get("target", ""))
            checkpoint = str(payload.get("checkpoint", ""))
            if not target or not checkpoint:
                raise ValueError(f"missing target or checkpoint metadata in {path}")
            previous_target = target_by_indicator.setdefault(indicator, target)
            previous_checkpoint = checkpoint_by_indicator.setdefault(
                indicator, checkpoint
            )
            if target != previous_target:
                raise ValueError(f"target changed between seeds in {path}")
            if checkpoint != previous_checkpoint:
                raise ValueError(f"checkpoint changed between seeds in {path}")

            transform = str(payload.get("transform", ""))
            if reference_transform is None:
                reference_transform = transform
            elif transform != reference_transform:
                raise ValueError(f"transform mode differs in {path}")
            source_parquet_value = payload.get("source_parquet")
            source_parquet = (
                None if source_parquet_value is None else str(source_parquet_value)
            )
            if not parquet_initialized:
                reference_parquet = source_parquet
                parquet_initialized = True
            elif source_parquet != reference_parquet:
                raise ValueError(f"source_parquet differs in {path}")

        seed_matrices.append(torch.stack(columns, dim=1))

    if canonical_rows is None or reference_transform is None:
        raise RuntimeError("no predictions were loaded")
    return LoadedPredictions(
        logits=torch.stack(seed_matrices, dim=0),
        source_rows=canonical_rows,
        targets=tuple(target_by_indicator[name] for name in layout.indicators),
        transform=reference_transform,
        source_parquet=reference_parquet,
        checkpoints={name: checkpoint_by_indicator[name] for name in layout.indicators},
    )


def reduce_pca(
    logits: torch.Tensor, n_components: int, fit_scope: str
) -> ReducedPredictions:
    if logits.ndim != 3:
        raise ValueError("combined logits must have shape [seeds, patients, indicators]")
    seed_count, patient_count, indicator_count = logits.shape
    if n_components <= 0 or n_components > indicator_count:
        raise ValueError(
            f"--n-components must be between 1 and {indicator_count}, got {n_components}"
        )
    if n_components > patient_count:
        raise ValueError(
            f"--n-components={n_components} exceeds patient count {patient_count}"
        )

    numpy_logits = logits.numpy()
    if fit_scope == "per-seed":
        values = []
        components = []
        feature_means = []
        explained_variances = []
        explained_ratios = []
        for seed_index in range(seed_count):
            reducer = PCA(n_components=n_components, random_state=0)
            values.append(reducer.fit_transform(numpy_logits[seed_index]))
            components.append(reducer.components_)
            feature_means.append(reducer.mean_)
            explained_variances.append(reducer.explained_variance_)
            explained_ratios.append(reducer.explained_variance_ratio_)
        return ReducedPredictions(
            values=torch.from_numpy(np.stack(values)).float(),
            components=torch.from_numpy(np.stack(components)).float(),
            feature_mean=torch.from_numpy(np.stack(feature_means)).float(),
            explained_variance=torch.from_numpy(
                np.stack(explained_variances)
            ).float(),
            explained_variance_ratio=torch.from_numpy(
                np.stack(explained_ratios)
            ).float(),
        )

    reducer = PCA(n_components=n_components, random_state=0)
    if fit_scope == "joint":
        fit_values = numpy_logits.reshape(seed_count * patient_count, indicator_count)
    elif fit_scope == "mean":
        fit_values = numpy_logits.mean(axis=0)
    else:
        raise ValueError(f"unknown PCA fit scope: {fit_scope!r}")
    reducer.fit(fit_values)
    transformed = reducer.transform(
        numpy_logits.reshape(seed_count * patient_count, indicator_count)
    ).reshape(seed_count, patient_count, n_components)
    return ReducedPredictions(
        values=torch.from_numpy(transformed).float(),
        components=torch.from_numpy(reducer.components_).float(),
        feature_mean=torch.from_numpy(reducer.mean_).float(),
        explained_variance=torch.from_numpy(reducer.explained_variance_).float(),
        explained_variance_ratio=torch.from_numpy(
            reducer.explained_variance_ratio_
        ).float(),
    )


def atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def run(args: argparse.Namespace) -> Path:
    layout = discover_layout(
        args.input_dir,
        indicators=args.indicators,
        split=args.split,
        seeds=args.seeds,
    )
    loaded = load_prediction_cube(layout, args.eye)
    if args.method != "pca":
        raise ValueError(f"unknown reduction method: {args.method!r}")
    reduced = reduce_pca(loaded.logits, args.n_components, args.pca_fit)

    output_file = args.output_file.resolve()
    atomic_save(
        {
            "X_prime": reduced.values,
            "source_row": loaded.source_rows,
            "seeds": torch.tensor(layout.seeds, dtype=torch.int64),
            "indicators": layout.indicators,
            "targets": loaded.targets,
            "eye": args.eye,
            "split": layout.split,
            "transform": loaded.transform,
            "source_parquet": loaded.source_parquet,
            "checkpoints": loaded.checkpoints,
            "reducer": {
                "method": args.method,
                "fit_scope": args.pca_fit,
                "n_components": args.n_components,
                "components": reduced.components,
                "feature_mean": reduced.feature_mean,
                "explained_variance": reduced.explained_variance,
                "explained_variance_ratio": reduced.explained_variance_ratio,
            },
        },
        output_file,
    )
    print(
        f"saved X_prime={tuple(reduced.values.shape)} eye={args.eye} "
        f"fit={args.pca_fit} output={output_file}",
        flush=True,
    )
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
