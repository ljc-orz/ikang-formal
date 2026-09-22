#!/usr/bin/env python3
"""Schedule split heatmap jobs from a text file across visible GPUs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.schedule_train import load_jobs, run  # noqa: E402


HEATMAP_SCRIPT = Path(__file__).with_name("generate_split_heatmaps.py")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "queue_file",
        type=Path,
        help="Text file containing one generate_split_heatmaps.py argument set per line",
    )
    parser.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent heatmap processes per visible GPU (default: 1)",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(
        parse_args(argv),
        program=HEATMAP_SCRIPT,
        default_log_root=REPO_DIR / "scheduler_logs" / "heatmap",
        process_kind="heatmap generation",
        job_loader=load_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
