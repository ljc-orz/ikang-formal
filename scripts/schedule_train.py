#!/usr/bin/env python3
"""Run queued train.py argument sets across every visible GPU."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Sequence

import torch


REPO_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_DIR / "train.py"


@dataclass(frozen=True)
class Job:
    number: int
    arguments: tuple[str, ...]


@dataclass
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen[str]
    log_stream: IO[str]
    log_path: Path
    started_at: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "queue_file",
        type=Path,
        help="Text file containing one train.py argument set per line",
    )
    parser.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent training processes on each visible GPU (default: 1)",
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
        help="Scheduler stdout/stderr directory (default: scheduler_logs/<time>)",
    )
    return parser.parse_args(argv)


def load_jobs(path: Path) -> list[Job]:
    resolved = path.resolve(strict=True)
    jobs: list[Job] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            arguments = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"{resolved}:{line_number}: {exc}") from exc
        if not arguments:
            continue
        if any(
            value == "--device" or value.startswith("--device=")
            for value in arguments
        ):
            raise ValueError(
                f"{resolved}:{line_number}: --device is assigned by the scheduler"
            )
        jobs.append(Job(number=line_number, arguments=tuple(arguments)))
    if not jobs:
        raise ValueError(f"queue file contains no jobs: {resolved}")
    return jobs


def visible_gpu_count() -> int:
    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("no CUDA GPU is visible to PyTorch")
    return count


def choose_gpu(active_counts: list[int], limit: int) -> int | None:
    candidates = [
        (count, gpu) for gpu, count in enumerate(active_counts) if count < limit
    ]
    return min(candidates)[1] if candidates else None


def terminate_jobs(running: list[RunningJob]) -> None:
    for item in running:
        if item.process.poll() is None:
            item.process.terminate()
    for item in running:
        try:
            item.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait()
        item.log_stream.close()


def run(args: argparse.Namespace) -> int:
    if args.max_processes_per_gpu <= 0:
        raise ValueError("--max-processes-per-gpu must be positive")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be positive")

    pending = deque(load_jobs(args.queue_file))
    gpu_count = visible_gpu_count()
    active_counts = [0] * gpu_count
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        args.log_dir.resolve()
        if args.log_dir is not None
        else (REPO_DIR / "scheduler_logs" / timestamp)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[RunningJob] = []
    failures: list[tuple[Job, int, int, Path]] = []
    completed = 0
    total = len(pending)
    print(
        f"queued={total} visible_gpus={gpu_count} "
        f"max_processes_per_gpu={args.max_processes_per_gpu} logs={log_dir}",
        flush=True,
    )

    try:
        while pending or running:
            while pending:
                gpu = choose_gpu(active_counts, args.max_processes_per_gpu)
                if gpu is None:
                    break
                job = pending.popleft()
                log_path = log_dir / f"job-{job.number:04d}-gpu-{gpu}.log"
                log_stream = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    "-u",
                    str(TRAIN_SCRIPT),
                    *job.arguments,
                    "--device",
                    f"cuda:{gpu}",
                ]
                print(
                    f"start job={job.number} gpu={gpu} log={log_path} "
                    f"command={shlex.join(command)}",
                    flush=True,
                )
                process = subprocess.Popen(
                    command,
                    cwd=REPO_DIR,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(
                    RunningJob(
                        job=job,
                        gpu=gpu,
                        process=process,
                        log_stream=log_stream,
                        log_path=log_path,
                        started_at=time.monotonic(),
                    )
                )
                active_counts[gpu] += 1

            if not running:
                continue
            time.sleep(args.poll_interval)
            for item in list(running):
                return_code = item.process.poll()
                if return_code is None:
                    continue
                item.log_stream.close()
                running.remove(item)
                active_counts[item.gpu] -= 1
                completed += 1
                duration = time.monotonic() - item.started_at
                status = "ok" if return_code == 0 else f"failed({return_code})"
                print(
                    f"finish job={item.job.number} gpu={item.gpu} status={status} "
                    f"seconds={duration:.1f} progress={completed}/{total} "
                    f"log={item.log_path}",
                    flush=True,
                )
                if return_code != 0:
                    failures.append(
                        (item.job, item.gpu, return_code, item.log_path)
                    )
    except KeyboardInterrupt:
        print("interrupted; terminating running training processes", file=sys.stderr)
        terminate_jobs(running)
        return 130

    if failures:
        print(f"completed with {len(failures)} failed job(s):", file=sys.stderr)
        for job, gpu, return_code, log_path in failures:
            print(
                f"  job={job.number} gpu={gpu} returncode={return_code} log={log_path}",
                file=sys.stderr,
            )
        return 1
    print(f"all {total} jobs completed successfully", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
