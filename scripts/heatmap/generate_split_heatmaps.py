#!/usr/bin/env python3
"""Generate compact native-resolution heatmaps for both eyes in one split."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.data import (  # noqa: E402
    PairedFundusWebDataset,
    build_eval_transform,
    build_train_transform,
)
from src.heatmap_vis import generate_heatmaps_batch, load_fundus_checkpoint  # noqa: E402


FORMAT = "paired-fundus-native-heatmaps-v1"
STORAGE_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--transform",
        choices=("eval", "train"),
        default="eval",
        help="Deterministic evaluation resize, or seeded stochastic augmentation",
    )
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "dali"),
        help="Default: use the backend recorded in the checkpoint",
    )
    parser.add_argument(
        "--explanation-target",
        choices=("predicted", "abnormal"),
        default="predicted",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=tuple(STORAGE_DTYPES),
        default="float16",
        help="Heatmap storage type; float16 minimizes space (default: float16)",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "fp16"),
        default="fp16",
        help="CUDA forward precision (default: fp16; automatically off on CPU)",
    )
    parser.add_argument(
        "--patient-batch-size",
        type=int,
        default=1,
        help="Patients decoded per iteration; the model sees twice this many eyes",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, help="Smoke-test limit")
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest(data_dir: Path, split: str) -> dict[str, Any]:
    manifest_path = data_dir / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in manifest.get("splits", {}):
        raise ValueError(
            f"unknown split {split!r}; available: {list(manifest.get('splits', {}))}"
        )
    return manifest


def build_loader(
    args: argparse.Namespace,
    loaded: Any,
    device: torch.device,
    backend: str,
) -> Any:
    data = loaded.config["data"]
    if backend == "torchvision":
        transform_builder = (
            build_train_transform if args.transform == "train" else build_eval_transform
        )
        dataset = PairedFundusWebDataset(
            args.data_dir,
            args.split,
            loaded.target,
            identity="source_row",
            transform=transform_builder(loaded.image_size, loaded.mean, loaded.std),
            seed=args.seed,
            skip_missing_target=False,
        )
        workers = int(data["num_workers"] if args.num_workers is None else args.num_workers)
        if workers < 0:
            raise ValueError("--num-workers must be non-negative")
        generator = torch.Generator().manual_seed(args.seed)
        return DataLoader(
            dataset,
            batch_size=args.patient_batch_size,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=generator,
        )
    if backend == "dali":
        if device.type != "cuda":
            raise RuntimeError("the DALI data backend requires a CUDA device")
        from src.data.dali_webdataset import DaliFundusLoader

        device_id = device.index if device.index is not None else torch.cuda.current_device()
        threads = int(
            data.get("dali_num_threads", 4)
            if args.num_workers is None
            else args.num_workers
        )
        return DaliFundusLoader(
            args.data_dir,
            args.split,
            loaded.target,
            mode="pairs",
            batch_size=args.patient_batch_size,
            num_threads=threads,
            device_id=device_id,
            image_size=loaded.image_size,
            mean=loaded.mean,
            std=loaded.std,
            seed=args.seed,
            skip_missing_target=False,
            identity="source_row",
            augment=args.transform == "train",
            dont_use_mmap=bool(data.get("dali_dont_use_mmap", False)),
            prefetch_queue_depth=int(data.get("dali_prefetch_queue_depth", 2)),
        )
    raise ValueError(f"unknown data backend: {backend!r}")


def _source_rows(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu", dtype=torch.int64)
    return torch.tensor([int(value) for value in values], dtype=torch.int64)


def _allocate_heatmaps(
    first: dict[str, torch.Tensor], patient_capacity: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, values in first.items():
        if values.ndim != 3:
            raise RuntimeError(
                f"native heatmap {name!r} must be [eyes,H,W], got {tuple(values.shape)}"
            )
        output[name] = torch.empty(
            (patient_capacity, 2, *values.shape[-2:]), dtype=dtype
        )
    return output


def atomic_save(payload: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_file)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.patient_batch_size <= 0:
        raise ValueError("--patient-batch-size must be positive")
    if args.max_patients is not None and args.max_patients <= 0:
        raise ValueError("--max-patients must be positive")
    if args.output_file.suffix != ".pt":
        raise ValueError("--output-file must end in .pt")
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = args.data_dir.expanduser().resolve(strict=True)
    manifest = _load_manifest(data_dir, args.split)
    loaded = load_fundus_checkpoint(args.checkpoint, device)
    amp_dtype = (
        torch.float16 if device.type == "cuda" and args.amp == "fp16" else None
    )
    backend = args.data_backend or str(loaded.config["data"].get("backend", "torchvision"))
    loader = build_loader(args, loaded, device, backend)

    full_count = int(manifest["splits"][args.split])
    capacity = full_count if args.max_patients is None else min(full_count, args.max_patients)
    source_rows = torch.empty(capacity, dtype=torch.int64)
    labels = torch.empty(capacity, dtype=torch.int8)
    ages = torch.empty(capacity, dtype=torch.int16)
    sexes = torch.empty(capacity, dtype=torch.uint8)
    logits = torch.empty((capacity, 2), dtype=torch.float32)
    predictions = torch.empty((capacity, 2), dtype=torch.uint8)
    stored_heatmaps: dict[str, torch.Tensor] | None = None
    storage_dtype = STORAGE_DTYPES[args.storage_dtype]
    offset = 0
    started = time.perf_counter()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=2,
    )
    with progress:
        task_id = progress.add_task(
            f"{loaded.target} {args.split}", total=capacity, status="starting"
        )
        for left, right, batch_ages, batch_sexes, batch_labels, identities in loader:
            if offset >= capacity:
                break
            batch_count = min(int(left.shape[0]), capacity - offset)
            left = left[:batch_count].to(device, non_blocking=True)
            right = right[:batch_count].to(device, non_blocking=True)
            batch_ages = batch_ages[:batch_count].to(device, non_blocking=True)
            batch_sexes = batch_sexes[:batch_count].to(device, non_blocking=True)
            # [patient-left, patient-right, ...] keeps eye order explicit.
            eye_images = torch.stack((left, right), dim=1).flatten(0, 1)
            eye_ages = batch_ages.repeat_interleave(2)
            eye_sexes = batch_sexes.repeat_interleave(2)
            result = generate_heatmaps_batch(
                loaded.model,
                eye_images,
                eye_ages,
                eye_sexes,
                decision_threshold=loaded.threshold,
                explanation_target=args.explanation_target,
                native_resolution=True,
                amp_dtype=amp_dtype,
            )
            if stored_heatmaps is None:
                stored_heatmaps = _allocate_heatmaps(
                    result.heatmaps, capacity, storage_dtype
                )
            expected_names = set(stored_heatmaps)
            if set(result.heatmaps) != expected_names:
                raise RuntimeError("heatmap methods changed while iterating the split")

            destination = slice(offset, offset + batch_count)
            for name, values in result.heatmaps.items():
                stored_heatmaps[name][destination].copy_(
                    values.reshape(batch_count, 2, *values.shape[-2:]).to(storage_dtype)
                )
            logits[destination].copy_(result.logits.reshape(batch_count, 2))
            predictions[destination].copy_(
                result.predicted_labels.reshape(batch_count, 2)
            )
            source_rows[destination].copy_(_source_rows(identities)[:batch_count])
            labels[destination].copy_(
                batch_labels[:batch_count].detach().to(device="cpu", dtype=torch.int8)
            )
            ages[destination].copy_(
                batch_ages.detach().to(device="cpu", dtype=torch.int16)
            )
            sexes[destination].copy_(
                batch_sexes.detach().to(device="cpu", dtype=torch.uint8)
            )
            offset += batch_count
            progress.update(
                task_id,
                advance=batch_count,
                status=f"patients={offset} eyes={2 * offset}",
            )

    if offset != capacity or stored_heatmaps is None:
        raise RuntimeError(f"loader produced {offset} patients, expected {capacity}")
    if torch.unique(source_rows).numel() != offset:
        raise RuntimeError("split heatmaps contain duplicate source_row values")
    parquet_rows = int(manifest["source"]["rows"])
    if bool(((source_rows < 0) | (source_rows >= parquet_rows)).any()):
        raise RuntimeError(f"source_row is outside parquet range [0, {parquet_rows})")

    payload = {
        "format": FORMAT,
        "source_row": source_rows,
        "heatmaps": stored_heatmaps,
        "logits": logits,
        "predictions": predictions,
        "labels": labels,
        "age": ages,
        "sex": sexes,
        "eye_order": ("left", "right"),
        "target": loaded.target,
        "split": args.split,
        "seed": int(args.seed),
        "transform": args.transform,
        "data_backend": backend,
        "checkpoint": str(loaded.checkpoint_path),
        "source_parquet": manifest["source"].get("parquet"),
        "backbone": loaded.backbone_type,
        "decision_threshold": loaded.threshold,
        "explanation_target": args.explanation_target,
        "heatmap_normalization": "independent min-max per eye and method",
        "native_resolution": True,
        "storage_dtype": args.storage_dtype,
        "amp": "fp16" if amp_dtype is not None else "off",
        "image_size": loaded.image_size,
    }
    output_file = args.output_file.expanduser().resolve()
    atomic_save(payload, output_file)
    elapsed = time.perf_counter() - started
    size_mib = output_file.stat().st_size / (1024 * 1024)
    shapes = {name: tuple(value.shape) for name, value in stored_heatmaps.items()}
    print(
        f"saved patients={offset} eyes={2 * offset} heatmaps={shapes} "
        f"dtype={args.storage_dtype} size={size_mib:.2f}MiB "
        f"amp={'fp16' if amp_dtype is not None else 'off'} "
        f"seconds={elapsed:.1f} output={output_file}",
        flush=True,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
