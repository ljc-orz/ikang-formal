#!/usr/bin/env python3
"""Run seeded paired-eye inference and save compact raw-logit tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data import (
    PairedFundusWebDataset,
    build_eval_transform,
    build_train_transform,
)
from src.model import FundusClassifier
from src.training import infer_paired_logits
from train import resolve_amp_dtype, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "internal_validation", "external_validation"),
        default="external_validation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One output .pt file is produced for each seed",
    )
    parser.add_argument(
        "--transform",
        choices=("train", "eval"),
        default="train",
        help="Seeded stochastic train augmentation, or deterministic eval resize",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "dali"),
        help="Override the checkpoint image backend",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="Debug/smoke-test limit")
    return parser.parse_args()


def build_model(checkpoint: dict[str, Any], device: torch.device) -> FundusClassifier:
    config = checkpoint["config"]
    data = config["data"]
    model_config = config["model"]
    model = FundusClassifier(
        backbone_type=model_config.get("backbone", "convnext"),
        model_name=model_config["name"],
        pretrained=False,
        image_size=int(data["image_size"]),
        metadata_hidden_dim=model_config["metadata_hidden_dim"],
        classifier_dropout=model_config["classifier_dropout"],
        drop_path_rate=model_config["drop_path_rate"],
        trainable_last_n_blocks=model_config.get("trainable_last_n_blocks", 3),
        lora_last_n_blocks=model_config.get("lora_last_n_blocks", 2),
        lora_rank=model_config.get("lora_rank", 8),
        lora_alpha=model_config.get("lora_alpha", 16.0),
        lora_dropout=model_config.get("lora_dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def build_loader(
    *,
    data_dir: Path,
    split: str,
    target: str,
    config: dict[str, Any],
    backend: str,
    transform_mode: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Any:
    data = config["data"]
    if backend == "torchvision":
        transform_builder = (
            build_train_transform if transform_mode == "train" else build_eval_transform
        )
        dataset = PairedFundusWebDataset(
            data_dir,
            split,
            target,
            identity="source_row",
            transform=transform_builder(data["image_size"], data["mean"], data["std"]),
            skip_missing_target=False,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
    if backend == "dali":
        if device.type != "cuda":
            raise RuntimeError("the DALI data backend requires a CUDA device")
        from src.data.dali_webdataset import DaliFundusLoader

        device_id = device.index if device.index is not None else torch.cuda.current_device()
        return DaliFundusLoader(
            data_dir,
            split,
            target,
            mode="pairs",
            batch_size=batch_size,
            num_threads=int(data.get("dali_num_threads", 4)),
            device_id=device_id,
            image_size=int(data["image_size"]),
            mean=data["mean"],
            std=data["std"],
            seed=seed,
            skip_missing_target=False,
            identity="source_row",
            augment=transform_mode == "train",
            dont_use_mmap=bool(data.get("dali_dont_use_mmap", False)),
            prefetch_queue_depth=int(data.get("dali_prefetch_queue_depth", 2)),
        )
    raise ValueError(f"unknown data backend: {backend!r}")


def validate_source_rows(
    source_rows: torch.Tensor,
    *,
    manifest: dict[str, Any],
    split: str,
    complete: bool,
) -> None:
    if source_rows.ndim != 1 or source_rows.dtype != torch.int64:
        raise RuntimeError("source_row must be a one-dimensional int64 tensor")
    if source_rows.numel() != torch.unique(source_rows).numel():
        raise RuntimeError("inference produced duplicate parquet source rows")
    parquet_rows = int(manifest["source"]["rows"])
    if bool(((source_rows < 0) | (source_rows >= parquet_rows)).any()):
        raise RuntimeError(f"source_row is outside the parquet range [0, {parquet_rows})")
    if complete and source_rows.numel() != int(manifest["splits"][split]):
        raise RuntimeError(
            f"inference returned {source_rows.numel()} patients, expected "
            f"{manifest['splits'][split]} for split {split!r}"
        )


def atomic_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")

    checkpoint_path = args.checkpoint.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    manifest = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    if args.split not in manifest.get("splits", {}):
        raise ValueError(f"dataset has no split {args.split!r}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target = str(checkpoint["target"])
    data = config["data"]
    backend = args.data_backend or str(data.get("backend", "torchvision"))
    batch_size = int(
        config["training"]["batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    num_workers = int(
        data["num_workers"] if args.num_workers is None else args.num_workers
    )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch size must be positive and workers must be non-negative")

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(config["training"]["amp"], device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows: torch.Tensor | None = None

    for seed in args.seeds:
        set_seed(seed)
        loader = build_loader(
            data_dir=data_dir,
            split=args.split,
            target=target,
            config=config,
            backend=backend,
            transform_mode=args.transform,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        source_rows, logits = infer_paired_logits(
            model,
            loader,
            device,
            amp_dtype,
            max_batches=args.max_batches,
            progress_desc=f"{target} {args.split} seed={seed}",
        )
        validate_source_rows(
            source_rows,
            manifest=manifest,
            split=args.split,
            complete=args.max_batches is None,
        )
        if reference_rows is None:
            reference_rows = source_rows
        elif not torch.equal(reference_rows, source_rows):
            raise RuntimeError("patient order changed between inference seeds")
        if logits.shape != (source_rows.numel(), 2):
            raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")

        output = args.output_dir / f"{target}.{args.split}.seed-{seed}.pt"
        atomic_save(
            {
                "source_row": source_rows,
                "logits": logits,
                "eye_order": ("left", "right"),
                "seed": seed,
                "target": target,
                "split": args.split,
                "transform": args.transform,
                "checkpoint": str(checkpoint_path),
                "source_parquet": manifest["source"].get("parquet"),
            },
            output,
        )
        print(
            f"saved seed={seed} patients={source_rows.numel()} "
            f"logits={tuple(logits.shape)} output={output.resolve()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
