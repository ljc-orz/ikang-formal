from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_prepare.summarize_split_prevalence import (
    summarize_parquet,
    write_csv,
)


class SplitPrevalenceTests(unittest.TestCase):
    def test_summarizes_each_split_and_excludes_missing_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "patients.parquet"
            pd.DataFrame(
                {
                    "uid": ["a", "b", "c", "d", "e", "f"],
                    "split": [
                        "train",
                        "train",
                        "train",
                        "internal_validation",
                        "internal_validation",
                        "external_validation",
                    ],
                    "result_alt": [0, 1, None, 1, 1, 0],
                    "result_bmi": [1, 1, 0, 0, None, 1],
                }
            ).to_parquet(parquet, index=False)

            split_rows, details, indicators = summarize_parquet(parquet)

        self.assertEqual(indicators, ["result_alt", "result_bmi"])
        self.assertEqual(
            [(row["split"], row["patients"]) for row in split_rows],
            [
                ("train", 3),
                ("internal_validation", 2),
                ("external_validation", 1),
            ],
        )
        by_key = {(row["split"], row["indicator"]): row for row in details}
        train_alt = by_key[("train", "result_alt")]
        self.assertEqual(train_alt["labeled_patients"], 2)
        self.assertEqual(train_alt["missing_patients"], 1)
        self.assertEqual(train_alt["abnormal_patients"], 1)
        self.assertEqual(train_alt["abnormal_prevalence"], 0.5)
        self.assertAlmostEqual(
            by_key[("train", "result_bmi")]["abnormal_prevalence"], 2 / 3
        )

    def test_rejects_non_binary_indicator_values(self) -> None:
        with TemporaryDirectory() as directory:
            parquet = Path(directory) / "invalid.parquet"
            pd.DataFrame(
                {"split": ["train"], "result_alt": [2]}
            ).to_parquet(parquet, index=False)
            with self.assertRaisesRegex(ValueError, "result_alt"):
                summarize_parquet(parquet)

    def test_writes_machine_readable_csv(self) -> None:
        rows = [
            {
                "split": "train",
                "split_patients": 3,
                "indicator": "result_alt",
                "labeled_patients": 2,
                "missing_patients": 1,
                "normal_patients": 1,
                "abnormal_patients": 1,
                "abnormal_prevalence": 0.5,
            }
        ]
        with TemporaryDirectory() as directory:
            output = write_csv(Path(directory) / "stats.csv", rows)
            with output.open(encoding="utf-8", newline="") as stream:
                loaded = list(csv.DictReader(stream))
        self.assertEqual(loaded[0]["split"], "train")
        self.assertEqual(loaded[0]["abnormal_prevalence"], "0.5")


if __name__ == "__main__":
    unittest.main()
