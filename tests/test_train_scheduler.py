from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.schedule_train import choose_gpu, load_jobs


class TrainSchedulerTests(unittest.TestCase):
    def test_load_jobs_parses_shell_quoting_and_comments(self) -> None:
        with TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text(
                "# comment\n"
                "--data-dir '/data/a b' --targets result_alt "
                "--output-dir runs/alt # trailing comment\n",
                encoding="utf-8",
            )
            jobs = load_jobs(queue)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].number, 2)
        self.assertEqual(jobs[0].arguments[1], "/data/a b")

    def test_load_jobs_rejects_device_owned_by_scheduler(self) -> None:
        with TemporaryDirectory() as directory:
            queue = Path(directory) / "jobs.txt"
            queue.write_text("--device=cuda:0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assigned by the scheduler"):
                load_jobs(queue)

    def test_choose_gpu_uses_least_loaded_available_gpu(self) -> None:
        self.assertEqual(choose_gpu([2, 0, 1, 0], limit=2), 1)
        self.assertIsNone(choose_gpu([2, 2], limit=2))


if __name__ == "__main__":
    unittest.main()
