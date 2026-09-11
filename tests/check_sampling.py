#!/usr/bin/env python3
"""Check the training-label distribution over multiple complete epochs."""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from train import (
    apply_cli_overrides,
    build_training_loader,
    parse_args,
    resolve_device,
    set_seed,
)


def expected_eye_counts(counts: dict[int, int], sampling: str) -> dict[int, int]:
    if sampling == "balanced":
        eyes_per_class = 2 * math.ceil((counts[0] + counts[1]) / 2)
        return {0: eyes_per_class, 1: eyes_per_class}
    return {0: 2 * counts[0], 1: 2 * counts[1]}


def inspect_epoch(
    loader: Any,
    *,
    target: str,
    epoch: int,
    sampling: str,
    expected: dict[int, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    totals = Counter({0: 0, 1: 0})
    distributions: Counter[tuple[int, int]] = Counter()

    for batch_index, batch in enumerate(loader, start=1):
        labels = batch[-1].detach().to(device="cpu", dtype=torch.int64)
        invalid = labels[(labels != 0) & (labels != 1)]
        if invalid.numel():
            raise AssertionError(
                f"{target} epoch {epoch} batch {batch_index} contains "
                f"non-binary labels: {invalid.unique().tolist()}"
            )
        batch_counts = torch.bincount(labels, minlength=2)
        count_0, count_1 = (int(value) for value in batch_counts.tolist())
        totals[0] += count_0
        totals[1] += count_1
        distributions[(count_0, count_1)] += 1
        if sampling == "balanced" and count_0 != count_1:
            raise AssertionError(
                f"{target} epoch {epoch} batch {batch_index} is not balanced: "
                f"count_0={count_0}, count_1={count_1}"
            )

    actual = {0: totals[0], 1: totals[1]}
    if actual != expected:
        raise AssertionError(
            f"{target} epoch {epoch} has unexpected label totals: "
            f"actual={actual}, expected={expected}"
        )

    total = totals[0] + totals[1]
    row = {
        "target": target,
        "epoch": epoch,
        "sampling": sampling,
        "count_0": totals[0],
        "count_1": totals[1],
        "ratio_0_1": totals[0] / totals[1],
        "positive_fraction": totals[1] / total,
        "total_eye_samples": total,
        "batch_count": sum(distributions.values()),
        "batch_distributions": [
            {"count_0": values[0], "count_1": values[1], "batches": frequency}
            for values, frequency in sorted(distributions.items())
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    print(
        f"[{target}] epoch={epoch} sampling={sampling} "
        f"count_0={totals[0]} count_1={totals[1]} "
        f"ratio_0_1={row['ratio_0_1']:.6f} "
        f"positive_fraction={row['positive_fraction']:.2%} "
        f"batches={row['batch_count']} "
        f"batch_distributions={row['batch_distributions']} "
        f"seconds={row['elapsed_seconds']:.2f} PASS",
        flush=True,
    )
    return row


def main() -> int:
    args = parse_args()
    if args.max_train_batches is not None:
        raise ValueError(
            "sampling checks require complete epochs; remove --max-train-batches"
        )

    config = apply_cli_overrides(load_config(args.config), args)
    epochs = int(config["training"]["max_epochs"])
    if epochs <= 0:
        raise ValueError("training.max_epochs must be positive")

    args.data_dir = args.data_dir.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    results: list[dict[str, Any]] = []

    for target in args.targets:
        if not target.startswith("result_") or Path(target).name != target:
            raise ValueError(f"invalid target name: {target!r}")
        set_seed(int(config["seed"]))
        loader, counts, backend, sampling = build_training_loader(
            target, config, args.data_dir, device
        )
        expected = expected_eye_counts(counts, sampling)
        print(
            f"[{target}] backend={backend} sampling={sampling} epochs={epochs} "
            f"patient_counts={counts} expected_eye_counts={expected}",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            results.append(
                inspect_epoch(
                    loader,
                    target=target,
                    epoch=epoch,
                    sampling=sampling,
                    expected=expected,
                )
            )

    report_path = args.output_dir / "sampling_check.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"all sampling checks passed; report={report_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
