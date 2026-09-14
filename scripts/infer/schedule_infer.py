#!/usr/bin/env python3
"""Run queued infer.py argument sets across every visible GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.schedule_train import run  # noqa: E402


INFER_SCRIPT = REPO_DIR / "infer.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "queue_file",
        type=Path,
        help="Text file containing one infer.py argument set per line",
    )
    parser.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent inference processes on each visible GPU (default: 1)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between process status checks (default: 1)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=(
            "Scheduler stdout/stderr directory "
            "(default: scheduler_logs/infer/<time>)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(
        parse_args(argv),
        program=INFER_SCRIPT,
        default_log_root=REPO_DIR / "scheduler_logs" / "infer",
        process_kind="inference",
    )


if __name__ == "__main__":
    raise SystemExit(main())
