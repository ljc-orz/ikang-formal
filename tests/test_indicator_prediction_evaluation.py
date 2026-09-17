from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from scripts.infer.evaluate_indicator_predictions import parse_args, run
from src.training import select_youden_threshold


def write_prediction_set(
    root: Path, parquet: Path, split: str, *, reverse: bool = False
) -> None:
    rows = [5, 2, 4, 1, 3, 0] if reverse else [0, 1, 2, 3, 4, 5]
    base = {
        "alt": torch.tensor([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]),
        "bmi": torch.tensor([-2.0, -0.2, -1.0, 0.4, 2.0, 1.2]),
    }
    for indicator, patient_logits in base.items():
        directory = root / indicator
        directory.mkdir(parents=True, exist_ok=True)
        for seed in (2026, 2027):
            offset = 0.1 * (seed - 2026)
            ordered = patient_logits[rows] + offset
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
                        "--output-dir",
                        str(output),
                        "--seeds",
                        "seq",
                        "2026",
                        "2027",
                        "--bootstrap-samples",
                        "30",
                        "--bootstrap-seed",
                        "11",
                    ]
                )
            )
            with (output / "indicator_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            summary = json.loads((output / "indicator_metrics.json").read_text())

        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["seeds"], [2026, 2027])
        self.assertEqual(summary["confidence_interval"]["samples"], 30)
        self.assertEqual(
            summary["confidence_interval"]["method"],
            "nonparametric bootstrap over TTA seeds",
        )
        by_key = {(row["indicator"], row["split"]): row for row in rows}
        self.assertEqual(
            by_key[("alt", "internal_validation")]["threshold"],
            by_key[("alt", "external_validation")]["threshold"],
        )
        base = torch.tensor([-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])
        expected_probabilities = torch.stack(
            [
                torch.stack((base + offset - 0.2, base + offset + 0.2), dim=1)
                .sigmoid()
                .mean(dim=1)
                for offset in (0.0, 0.1)
            ]
        ).mean(dim=0).numpy()
        expected_threshold = select_youden_threshold(
            np.array([0, 0, 0, 1, 1, 1]), expected_probabilities
        )
        self.assertAlmostEqual(
            float(by_key[("alt", "internal_validation")]["threshold"]),
            expected_threshold,
        )
        self.assertEqual(float(by_key[("alt", "external_validation")]["auroc"]), 1.0)
        self.assertEqual(
            float(by_key[("alt", "external_validation")]["auroc_ci_low"]), 1.0
        )
        self.assertEqual(
            float(by_key[("alt", "external_validation")]["auroc_ci_high"]), 1.0
        )


if __name__ == "__main__":
    unittest.main()
