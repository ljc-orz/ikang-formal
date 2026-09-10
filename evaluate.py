#!/usr/bin/env python3
"""Evaluate a checkpoint by predicting both eyes and averaging probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import PairedFundusWebDataset, build_eval_transform, count_target_values
from src.model import FundusClassifier
from src.training import evaluate_paired_eyes
from src.utils import json_safe
from train import make_criterion, resolve_amp_dtype, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "internal_validation", "external_validation"),
        default="external_validation",
    )
    parser.add_argument("--output", type=Path, help="Prediction CSV path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "dali"),
        help="Override the checkpoint image data backend",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, help="Default: checkpoint threshold")
    parser.add_argument("--max-batches", type=int, help="Debug/smoke-test limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_started = time.perf_counter()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    target = checkpoint["target"]
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(config["training"]["amp"], device)
    data = config["data"]
    data_backend = args.data_backend or data.get("backend", "torchvision")
    batch_size = (
        config["training"]["batch_size"] if args.batch_size is None else args.batch_size
    )
    if data_backend == "torchvision":
        transform = build_eval_transform(data["image_size"], data["mean"], data["std"])
        dataset = PairedFundusWebDataset(
            args.data_dir,
            args.split,
            target,
            transform=transform,
            skip_missing_target=False,
        )
        workers = data["num_workers"] if args.num_workers is None else args.num_workers
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
    elif data_backend == "dali":
        if device.type != "cuda":
            raise RuntimeError("the DALI data backend requires a CUDA device")
        from src.data.dali_webdataset import DaliFundusLoader

        device_id = device.index if device.index is not None else torch.cuda.current_device()
        loader = DaliFundusLoader(
            args.data_dir,
            args.split,
            target,
            mode="pairs",
            batch_size=int(batch_size),
            num_threads=int(data.get("dali_num_threads", 4)),
            device_id=device_id,
            image_size=int(data["image_size"]),
            mean=data["mean"],
            std=data["std"],
            seed=int(config["seed"]),
            skip_missing_target=False,
            dont_use_mmap=bool(data.get("dali_dont_use_mmap", False)),
            prefetch_queue_depth=int(data.get("dali_prefetch_queue_depth", 2)),
        )
    else:
        raise ValueError(f"unknown data backend: {data_backend!r}")
    model = FundusClassifier(
        backbone_type=config["model"].get("backbone", "convnext"),
        model_name=config["model"]["name"],
        pretrained=False,
        image_size=int(data["image_size"]),
        metadata_hidden_dim=config["model"]["metadata_hidden_dim"],
        classifier_dropout=config["model"]["classifier_dropout"],
        drop_path_rate=config["model"]["drop_path_rate"],
        lora_last_n_blocks=config["model"].get("lora_last_n_blocks", 2),
        lora_rank=config["model"].get("lora_rank", 8),
        lora_alpha=config["model"].get("lora_alpha", 16.0),
        lora_dropout=config["model"].get("lora_dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    # The loss is reported for labeled patients; missing targets remain in the CSV.
    train_counts = count_target_values(args.data_dir, data["train_split"], target)
    criterion, _ = make_criterion(config, train_counts, device)
    threshold = checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold
    metrics, predictions = evaluate_paired_eyes(
        model,
        loader,
        criterion,
        device,
        amp_dtype,
        threshold=threshold,
        max_batches=args.max_batches,
        progress_desc=f"{target} {args.split}",
    )

    output = args.output or checkpoint_path.with_name(f"{args.split}_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "patient_id",
                "target",
                "left_probability",
                "right_probability",
                "probability",
                "prediction",
            ]
        )
        for patient_id, target_value, left, right, probability in zip(
            predictions["patient_id"],
            predictions["target"],
            predictions["left_probability"],
            predictions["right_probability"],
            predictions["probability"],
        ):
            writer.writerow(
                [
                    patient_id,
                    int(target_value),
                    float(left),
                    float(right),
                    float(probability),
                    int(probability >= metrics["threshold"]),
                ]
            )
    metrics.update(
        {
            "target": target,
            "split": args.split,
            "data_backend": data_backend,
            "patients": len(predictions["target"]),
            "labeled_patients": int(np.sum(predictions["target"] >= 0)),
            "predictions": str(output.resolve()),
            "evaluation_seconds": time.perf_counter() - evaluation_started,
        }
    )
    metrics_path = output.with_suffix(".metrics.json")
    serialized_metrics = json_safe(metrics)
    metrics_path.write_text(
        json.dumps(serialized_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = output.with_suffix(".log")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{timestamp} evaluation_result="
            + json.dumps(serialized_metrics, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    print(json.dumps(serialized_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
