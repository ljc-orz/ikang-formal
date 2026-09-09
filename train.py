#!/usr/bin/env python3
"""Train one independent V1 binary classifier per selected indicator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import (
    FundusWebDataset,
    PairedFundusWebDataset,
    build_eval_transform,
    build_train_transform,
    count_target_values,
)
from src.model import FundusClassifier
from src.training import evaluate_paired_eyes, train_one_epoch
from src.training.optim import build_optimizer, build_scheduler
from src.utils import json_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True, help="result_* names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--model", help="Override the timm model name")
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        help="Override the local pretrained checkpoint path",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, help="Debug/smoke-test limit")
    parser.add_argument("--max-val-batches", type=int, help="Debug/smoke-test limit")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_amp_dtype(mode: str | bool, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or mode is False or str(mode).lower() in {"off", "false", "none"}:
        return None
    normalized = str(mode).lower()
    if normalized in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP was requested but this GPU does not support it")
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized != "auto":
        raise ValueError(f"unknown AMP mode: {mode!r}")
    # Keep automatic mode aligned with the project default. BF16 remains
    # available only when explicitly requested with training.amp: bf16.
    return torch.float16


def make_loader(dataset: torch.utils.data.IterableDataset, config: dict[str, Any]) -> DataLoader:
    workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_criterion(
    config: dict[str, Any], counts: dict[int, int], device: torch.device
) -> tuple[nn.Module, float]:
    setting = config["loss"]["pos_weight"]
    if setting == "auto":
        if counts[0] == 0 or counts[1] == 0:
            print(
                f"warning: training target contains only one class ({counts}); "
                "using pos_weight=1.0 (metrics such as AUROC will be undefined)"
            )
            weight = 1.0
        else:
            weight = min(
                counts[0] / counts[1], float(config["loss"]["max_pos_weight"])
            )
    elif setting is None or setting is False:
        weight = 1.0
    else:
        weight = float(setting)
        if weight <= 0:
            raise ValueError("loss.pos_weight must be positive")
    tensor = torch.tensor(weight, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=tensor), weight


def atomic_torch_save(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def log_message(path: Path, message: str) -> None:
    """Print and append a timestamped message to a durable per-target log."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def is_better(value: float, best: float, mode: str, min_delta: float) -> bool:
    if not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    return value > best + min_delta if mode == "max" else value < best - min_delta


def current_learning_rate(optimizer: torch.optim.Optimizer, scope: str) -> float:
    values = [
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("scope") == scope
    ]
    if not values:
        raise RuntimeError(f"optimizer has no {scope!r} parameter group")
    return max(values)


def train_target(
    target: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not target.startswith("result_") or Path(target).name != target:
        raise ValueError(f"invalid target name: {target!r}")
    set_seed(int(config["seed"]))
    target_dir = (args.output_dir / target).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "training.log"
    run_started = time.perf_counter()
    data = config["data"]
    training = config["training"]
    counts = count_target_values(args.data_dir, data["train_split"], target)
    validation_counts = count_target_values(
        args.data_dir, data["validation_split"], target
    )
    train_transform = build_train_transform(data["image_size"], data["mean"], data["std"])
    eval_transform = build_eval_transform(data["image_size"], data["mean"], data["std"])
    train_dataset = FundusWebDataset(
        args.data_dir,
        data["train_split"],
        target,
        transform=train_transform,
        shuffle=True,
        shuffle_buffer=data["shuffle_buffer"],
        seed=config["seed"],
        skip_missing_target=True,
    )
    validation_dataset = PairedFundusWebDataset(
        args.data_dir,
        data["validation_split"],
        target,
        transform=eval_transform,
        skip_missing_target=True,
    )
    train_loader = make_loader(train_dataset, config)
    validation_loader = make_loader(validation_dataset, config)

    model = FundusClassifier(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        pretrained_weights=config["model"]["pretrained_weights"],
        metadata_hidden_dim=config["model"]["metadata_hidden_dim"],
        classifier_dropout=config["model"]["classifier_dropout"],
        drop_path_rate=config["model"]["drop_path_rate"],
    ).to(device)
    criterion, pos_weight = make_criterion(config, counts, device)
    optimizer = build_optimizer(
        model,
        training["backbone_lr"],
        training["head_lr"],
        training["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        training["max_epochs"],
        config["scheduler"]["warmup_epochs"],
        config["scheduler"]["min_lr"],
    )
    amp_dtype = resolve_amp_dtype(training["amp"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    trainable, total = model.trainable_parameter_counts()
    batch_size = int(training["batch_size"])
    train_batches = math.ceil(2 * (counts[0] + counts[1]) / batch_size)
    validation_batches = math.ceil(
        (validation_counts[0] + validation_counts[1]) / batch_size
    )
    checkpoint_config = json.loads(json.dumps(config))
    (target_dir / "run_config.json").write_text(
        json.dumps(checkpoint_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_message(
        log_path,
        f"[{target}] start device={device} amp={amp_dtype} "
        f"train_counts={counts} validation_counts={validation_counts} "
        f"pos_weight={pos_weight:.6g} trainable={trainable:,}/{total:,} "
        f"batch_size={batch_size} accumulation_steps="
        f"{training['gradient_accumulation_steps']} output_dir={target_dir}",
    )
    log_message(
        log_path,
        f"[{target}] config="
        + json.dumps(checkpoint_config, ensure_ascii=False, sort_keys=True),
    )

    history: list[dict[str, Any]] = []
    primary = config["checkpoint"]["primary_metric"]
    secondary = config["checkpoint"]["secondary_metric"]
    mode = config["checkpoint"]["mode"]
    if mode not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be 'min' or 'max'")
    if primary not in {"val_auprc", "val_auroc", "val_sensitivity", "val_specificity", "val_loss"}:
        raise ValueError(f"unsupported checkpoint.primary_metric: {primary}")
    if secondary not in {
        "val_auprc",
        "val_auroc",
        "val_sensitivity",
        "val_specificity",
        "val_loss",
    }:
        raise ValueError(f"unsupported checkpoint.secondary_metric: {secondary}")
    best = -math.inf if mode == "max" else math.inf
    best_secondary = -math.inf if mode == "max" else math.inf
    patience = 0
    best_epoch = -1

    max_epochs = int(training["max_epochs"])
    for epoch in range(max_epochs):
        epoch_number = epoch + 1
        epoch_started = time.perf_counter()
        log_message(log_path, f"[{target}] epoch={epoch_number}/{max_epochs} train_start")
        train_started = time.perf_counter()
        train_values = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_dtype,
            int(training["gradient_accumulation_steps"]),
            float(training["gradient_clip_norm"]),
            args.max_train_batches,
            progress_desc=f"{target} train {epoch_number}/{max_epochs}",
            total_batches=train_batches,
        )
        train_seconds = time.perf_counter() - train_started
        log_message(
            log_path,
            f"[{target}] epoch={epoch_number}/{max_epochs} validation_start",
        )
        validation_started = time.perf_counter()
        validation_values, _ = evaluate_paired_eyes(
            model,
            validation_loader,
            criterion,
            device,
            amp_dtype,
            max_batches=args.max_val_batches,
            progress_desc=f"{target} validation {epoch_number}/{max_epochs}",
            total_batches=validation_batches,
        )
        validation_seconds = time.perf_counter() - validation_started
        row: dict[str, Any] = {"epoch": epoch_number}
        row.update(prefixed("train", train_values))
        row.update(prefixed("val", validation_values))
        row["backbone_lr"] = current_learning_rate(optimizer, "backbone")
        row["head_lr"] = current_learning_rate(optimizer, "head")
        row["train_seconds"] = train_seconds
        row["validation_seconds"] = validation_seconds
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        score = float(row[primary])
        secondary_score = float(row[secondary])
        primary_improved = is_better(
            score, best, mode, float(training["early_stopping_min_delta"])
        )
        primary_tied = (
            math.isfinite(score)
            and math.isfinite(best)
            and abs(score - best) <= float(training["early_stopping_min_delta"])
        )
        secondary_improved = primary_tied and is_better(
            secondary_score, best_secondary, mode, 0.0
        )
        improved = best_epoch < 0 or primary_improved or secondary_improved
        if improved:
            if best_epoch < 0 or primary_improved:
                best = score
            best_secondary = secondary_score
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        row["is_best"] = improved
        row["early_stopping_patience"] = patience
        row["best_epoch"] = best_epoch
        row["best_value"] = best
        history.append(row)
        write_history(target_dir / "history.csv", history)

        checkpoint = {
            "version": 1,
            "target": target,
            "epoch": epoch_number,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": checkpoint_config,
            "threshold": validation_values["threshold"],
            "metrics": row,
            "label_counts": counts,
        }
        log_message(
            log_path,
            f"[{target}] epoch={epoch_number}/{max_epochs} saving_checkpoints "
            f"is_best={improved}",
        )
        atomic_torch_save(checkpoint, target_dir / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, target_dir / "best.pt")
        log_message(
            log_path,
            f"[{target}] epoch_result="
            + json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True),
        )
        scheduler.step()
        if patience >= int(training["early_stopping_patience"]):
            log_message(log_path, f"[{target}] early_stopping epoch={epoch_number}")
            break

    summary = {
        "target": target,
        "best_epoch": best_epoch,
        "primary_metric": primary,
        "best_value": best,
        "secondary_metric": secondary,
        "best_secondary_value": best_secondary,
        "checkpoint": str(target_dir / "best.pt"),
        "epochs_completed": len(history),
        "total_seconds": time.perf_counter() - run_started,
    }
    (target_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_message(
        log_path,
        f"[{target}] complete summary="
        + json.dumps(json_safe(summary), ensure_ascii=False, sort_keys=True),
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.pretrained_weights is not None:
        config["model"]["pretrained_weights"] = str(args.pretrained_weights)
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if int(config["training"]["max_epochs"]) <= 0:
        raise ValueError("training.max_epochs must be positive")
    if int(config["training"]["gradient_accumulation_steps"]) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")

    args.data_dir = args.data_dir.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    summaries = [train_target(target, config, args, device) for target in args.targets]
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summaries), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
