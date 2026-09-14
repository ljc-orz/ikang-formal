from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.infer.schedule_infer import (
    INFER_SCRIPT,
    REPO_DIR,
    expand_seed_sequence,
    load_infer_jobs,
    parse_args,
)
from scripts.schedule_train import Job, build_command, load_jobs


class InferSchedulerTests(unittest.TestCase):
    def test_queue_supports_infer_arguments_and_comments(self) -> None:
        with TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text(
                "# infer two checkpoints in parallel\n"
                "--checkpoint 'runs/model one/best.pt' --data-dir /data/fundus "
                "--output-dir runs/model-one/infer --seeds 7 11 # random TTA\n",
                encoding="utf-8",
            )
            jobs = load_jobs(queue)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].number, 2)
        self.assertIn("runs/model one/best.pt", jobs[0].arguments)
        self.assertEqual(jobs[0].arguments[-2:], ("7", "11"))

    def test_command_runs_infer_and_scheduler_owns_device(self) -> None:
        job = Job(number=1, arguments=("--checkpoint", "best.pt", "--seeds", "3"))
        command = build_command(INFER_SCRIPT, job, gpu=2)

        self.assertEqual(command[:3], [sys.executable, "-u", str(REPO_DIR / "infer.py")])
        self.assertEqual(command[-2:], ["--device", "cuda:2"])

    def test_seed_sequence_is_inclusive(self) -> None:
        job = Job(
            number=4,
            arguments=(
                "--checkpoint",
                "best.pt",
                "--seeds",
                "seq",
                "2026",
                "2029",
                "--transform",
                "train",
            ),
        )

        expanded = expand_seed_sequence(job)

        self.assertEqual(
            expanded.arguments[3:7], ("2026", "2027", "2028", "2029")
        )
        self.assertEqual(expanded.arguments[-2:], ("--transform", "train"))

    def test_infer_job_loader_reports_invalid_seed_sequence_line(self) -> None:
        with TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text(
                "# comment\n--checkpoint best.pt --seeds seq 9 3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r":2:.*FIRST <= LAST"):
                load_infer_jobs(queue)

    def test_explicit_seed_list_is_unchanged(self) -> None:
        job = Job(number=1, arguments=("--seeds", "3", "5", "8"))

        self.assertIs(expand_seed_sequence(job), job)

    def test_infer_scheduler_defaults(self) -> None:
        args = parse_args(["jobs.txt"])

        self.assertEqual(args.queue_file, Path("jobs.txt"))
        self.assertEqual(args.max_processes_per_gpu, 1)
        self.assertEqual(args.poll_interval, 1.0)
        self.assertIsNone(args.log_dir)


if __name__ == "__main__":
    unittest.main()
