"""One-epoch training and paired-eye evaluation loops."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
from typing import Any

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import nn

from .metrics import binary_metrics


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float,
    gradient_scale: float = 1.0,
) -> None:
    scaler.unscale_(optimizer)
    if gradient_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _progress_batches(
    loader: torch.utils.data.DataLoader,
    description: str | None,
    max_batches: int | None,
    total_batches: int | None,
):
    batches = loader if max_batches is None else islice(loader, max_batches)
    if total_batches is None:
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=5,
        disable=description is None,
    )
    with progress:
        task_id = progress.add_task(
            description or "processing",
            total=total_batches,
            status="starting",
        )
        for batch in batches:
            yield progress, task_id, batch


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    accumulation_steps: int,
    gradient_clip_norm: float,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    total_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    pending = 0

    progress_batches = _progress_batches(loader, progress_desc, max_batches, total_batches)
    for progress, task_id, (images, ages, sexes, labels) in progress_batches:
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        sexes = sexes.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            logits = model(images, ages, sexes)
            loss = criterion(logits, labels)
        scaler.scale(loss / accumulation_steps).backward()
        pending += 1
        if pending == accumulation_steps:
            _optimizer_step(model, optimizer, scaler, gradient_clip_norm)
            pending = 0

        batch_size = labels.numel()
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(logits.detach().float().sigmoid().cpu().numpy())
        progress.update(
            task_id,
            advance=1,
            status=(
                f"loss={loss_sum / sample_count:.5f} "
                f"samples={sample_count} lr={optimizer.param_groups[0]['lr']:.2e}"
            ),
        )

    if pending:
        _optimizer_step(
            model,
            optimizer,
            scaler,
            gradient_clip_norm,
            gradient_scale=accumulation_steps / pending,
        )
    if sample_count == 0:
        raise RuntimeError("training loader produced no labeled samples")
    metrics = binary_metrics(np.concatenate(targets), np.concatenate(probabilities))
    return {"loss": loss_sum / sample_count, **metrics}


@torch.inference_mode()
def evaluate_paired_eyes(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    threshold: float | None = None,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    total_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Predict each eye independently and calculate metrics from their mean."""
    model.eval()
    targets: list[np.ndarray] = []
    left_probabilities: list[np.ndarray] = []
    right_probabilities: list[np.ndarray] = []
    patient_ids: list[str] = []
    loss_sum = 0.0
    labeled_eye_count = 0

    progress_batches = _progress_batches(loader, progress_desc, max_batches, total_batches)
    patient_count = 0
    for progress, task_id, (left, right, ages, sexes, labels, ids) in progress_batches:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        sexes = sexes.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        valid = labels >= 0
        with _autocast(device, amp_dtype):
            left_logits = model(left, ages, sexes)
            right_logits = model(right, ages, sexes)
            if valid.any():
                loss = 0.5 * (
                    criterion(left_logits[valid], labels[valid])
                    + criterion(right_logits[valid], labels[valid])
                )
            else:
                loss = None
        batch_size = labels.numel()
        valid_count = int(valid.sum())
        if loss is not None:
            loss_sum += float(loss) * valid_count * 2
            labeled_eye_count += valid_count * 2
        targets.append(labels.cpu().numpy())
        left_probabilities.append(left_logits.float().sigmoid().cpu().numpy())
        right_probabilities.append(right_logits.float().sigmoid().cpu().numpy())
        patient_ids.extend(str(item) for item in ids)
        patient_count += batch_size
        running_loss = loss_sum / labeled_eye_count if labeled_eye_count else float("nan")
        progress.update(
            task_id,
            advance=1,
            status=f"loss={running_loss:.5f} patients={patient_count}",
        )

    if not targets:
        raise RuntimeError("evaluation loader produced no labeled patients")
    target_array = np.concatenate(targets).astype(np.int64)
    left_array = np.concatenate(left_probabilities)
    right_array = np.concatenate(right_probabilities)
    mean_array = (left_array + right_array) / 2.0
    valid_array = target_array >= 0
    if valid_array.any():
        metrics = binary_metrics(target_array[valid_array], mean_array[valid_array], threshold)
        metrics["loss"] = loss_sum / labeled_eye_count
    else:
        metrics = {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "threshold": 0.5 if threshold is None else float(threshold),
            "loss": float("nan"),
        }
    predictions = {
        "patient_id": patient_ids,
        "target": target_array,
        "left_probability": left_array,
        "right_probability": right_array,
        "probability": mean_array,
    }
    return metrics, predictions
