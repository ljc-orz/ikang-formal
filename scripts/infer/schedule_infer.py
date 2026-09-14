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

from scripts.schedule_train import Job, load_jobs, run  # noqa: E402


INFER_SCRIPT = REPO_DIR / "infer.py"


def expand_seed_sequence(job: Job) -> Job:
    """Expand ``--seeds seq FIRST LAST`` to an inclusive integer sequence."""
    arguments = list(job.arguments)
    try:
        seeds_index = arguments.index("--seeds")
    except ValueError:
        return job

    if seeds_index + 1 >= len(arguments) or arguments[seeds_index + 1] != "seq":
        return job
    if seeds_index + 3 >= len(arguments):
        raise ValueError("--seeds seq requires FIRST and LAST")

    try:
        first = int(arguments[seeds_index + 2])
        last = int(arguments[seeds_index + 3])
    except ValueError as exc:
        raise ValueError("--seeds seq FIRST LAST must use integers") from exc
    if first > last:
        raise ValueError("--seeds seq requires FIRST <= LAST")
    if seeds_index + 4 < len(arguments) and not arguments[seeds_index + 4].startswith(
        "--"
    ):
        raise ValueError("--seeds seq accepts exactly FIRST and LAST")

    arguments[seeds_index + 1 : seeds_index + 4] = [
        str(seed) for seed in range(first, last + 1)
    ]
    return Job(number=job.number, arguments=tuple(arguments))


def load_infer_jobs(path: Path) -> list[Job]:
    jobs = load_jobs(path)
    expanded: list[Job] = []
    for job in jobs:
        try:
            expanded.append(expand_seed_sequence(job))
        except ValueError as exc:
            raise ValueError(f"{path.resolve()}:{job.number}: {exc}") from exc
    return expanded


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
        job_loader=load_infer_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
