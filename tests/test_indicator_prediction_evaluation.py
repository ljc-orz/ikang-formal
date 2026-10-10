from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from scripts.infer.evaluate_indicator_predictions import parse_args, run


def write_prediction_set(
    root: Path, parquet: Path, split: str, *, reverse: bool = False
) -> None:
    rows = [5, 2, 4, 1, 3, 0] if reverse else [0, 1, 2, 3, 4, 5]
    base = {
        "alt": torch.tensor([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]),
        "bmi": torch.tensor([-2.0, -0.2, -1.0, 0.4, 2.0, 1.2]),
    }
    jitter = {
        "alt": torch.tensor([0.0, 0.0, 0.0, 0.0, -3.0, 0.0]),
        "bmi": torch.tensor([0.0, 0.0, 1.5, 0.0, -2.0, 0.0]),
    }
    for indicator, patient_logits in base.items():
        directory = root / indicator
        directory.mkdir(parents=True, exist_ok=True)
        for seed in (2026, 2027):
            seed_logits = patient_logits + jitter[indicator] * (seed - 2026)
            ordered = seed_logits[rows]
            logits = torch.stack((ordered - 0.2, ordered + 0.2), dim=1)
            torch.save(
                {
                    "source_row": torch.tensor(rows, dtype=torch.int64),
                    "logits": logits,
                    "eye_order": ("left", "right"),
                    "seed": seed,
                    "target": f"result_{indicator}",
                    "split": split,
                    "transform": "train",
                    "checkpoint": f"/models/{indicator}/best.pt",
                    "source_parquet": str(parquet),
                },
                directory / f"result_{indicator}.{split}.seed-{seed}.pt",
            )


class IndicatorPredictionEvaluationTests(unittest.TestCase):
    def test_evaluates_tta_ensemble_and_reuses_internal_threshold(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "patients.parquet"
            pd.DataFrame(
                {
                    "result_alt": [0, 0, 0, 1, 1, 1],
                    "result_bmi": [0, 0, 1, 0, 1, 1],
                }
            ).to_parquet(parquet)
            internal = root / "internal"
            external = root / "external"
            train = root / "train"
            write_prediction_set(train, parquet, "train")
            write_prediction_set(internal, parquet, "internal_validation")
            write_prediction_set(
                external, parquet, "external_validation", reverse=True
            )
            output = root / "metrics"
            run(
                parse_args(
                    [
                        "--internal-predictions-dir",
                        str(internal),
                        "--external-predictions-dir",
                        str(external),
                        "--train-predictions-dir",
                        str(train),
                        "--output-dir",
                        str(output),
                        "--seeds",
                        "seq",
                        "2026",
                        "2027",
                    ]
                )
            )
            with (output / "indicator_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            with (output / "indicator_seed_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                seed_rows = list(csv.DictReader(stream))
            summary = json.loads((output / "indicator_metrics.json").read_text())

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(seed_rows), 12)
        self.assertEqual(summary["seeds"], [2026, 2027])
        self.assertEqual(
            summary["splits"],
            ["train", "internal_validation", "external_validation"],
        )
        self.assertEqual(summary["confidence_interval"]["degrees_of_freedom"], 1)
        self.assertEqual(
            summary["confidence_interval"]["method"],
            "two-sided Student t interval over per-seed metric values",
        )
        by_key = {(row["indicator"], row["split"]): row for row in rows}
        self.assertEqual(
            by_key[("alt", "internal_validation")]["threshold"],
            by_key[("alt", "external_validation")]["threshold"],
        )
        self.assertEqual(
            by_key[("alt", "internal_validation")]["threshold"],
            by_key[("alt", "train")]["threshold"],
        )
        alt_external_seed_aurocs = np.asarray(
            [
                float(row["auroc"])
                for row in seed_rows
                if row["indicator"] == "alt"
                and row["split"] == "external_validation"
            ]
        )
        expected_mean = float(alt_external_seed_aurocs.mean())
        expected_std = float(alt_external_seed_aurocs.std(ddof=1))
        expected_margin = float(student_t.ppf(0.975, 1)) * expected_std / np.sqrt(2)
        result = by_key[("alt", "external_validation")]
        self.assertAlmostEqual(float(result["auroc_mean"]), expected_mean)
        self.assertAlmostEqual(float(result["auroc_std"]), expected_std)
        self.assertAlmostEqual(
            float(result["auroc_ci_low"]), expected_mean - expected_margin
        )
        self.assertAlmostEqual(
            float(result["auroc_ci_high"]), expected_mean + expected_margin
        )


if __name__ == "__main__":
    unittest.main()
