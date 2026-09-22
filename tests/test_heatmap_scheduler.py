from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.heatmap.generate_split_heatmaps import parse_args as parse_generate_args
from scripts.heatmap.schedule_heatmaps import HEATMAP_SCRIPT, parse_args
from scripts.schedule_train import load_jobs


class HeatmapSchedulerTests(unittest.TestCase):
    def test_queue_supports_comments_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text(
                "# comment\n"
                "--checkpoint 'run one/best.pt' --data-dir data --split test "
                "--output-file out.pt --seed 42 # trailing comment\n",
                encoding="utf-8",
            )
            jobs = load_jobs(queue)
        self.assertEqual(len(jobs), 1)
        self.assertIn("run one/best.pt", jobs[0].arguments)
        self.assertEqual(jobs[0].arguments[-2:], ("--seed", "42"))

    def test_queue_rejects_scheduler_owned_device(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text("--checkpoint x --device cuda:0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assigned by the scheduler"):
                load_jobs(queue)

    def test_scheduler_defaults(self):
        arguments = parse_args(["jobs.txt"])
        self.assertEqual(arguments.max_processes_per_gpu, 1)
        self.assertEqual(arguments.poll_interval, 1.0)
        self.assertTrue(HEATMAP_SCRIPT.is_file())

    def test_generator_defaults_to_fp16_amp(self):
        arguments = parse_generate_args(
            [
                "--checkpoint",
                "best.pt",
                "--data-dir",
                "webdataset",
                "--split",
                "external_validation",
                "--output-file",
                "heatmaps.pt",
            ]
        )
        self.assertEqual(arguments.amp, "fp16")


if __name__ == "__main__":
    unittest.main()
