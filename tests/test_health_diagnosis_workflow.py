from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from scripts.infer.fit_health_diagnosis import parse_args, run
from src.health_diagnosis import HealthDiagnosticModel, load_aggregated_predictions


def write_aggregate(path: Path, parquet: Path, split: str) -> None:
    generator = torch.Generator().manual_seed(31)
    seeds = torch.tensor([2026, 2027, 2028, 2029, 2030], dtype=torch.int64)
    healthy = torch.randn(5, 4, 2, generator=generator) * 0.1
    abnormal = 3.0 + torch.randn(5, 4, 2, generator=generator) * 0.1
    unlabeled = torch.randn(5, 1, 2, generator=generator)
    logits = torch.cat((healthy, abnormal, unlabeled), dim=1)
    torch.save(
        {
            "X": logits,
            "X_prime": logits[..., :1],
            "left_right_abs_difference": torch.rand(
                logits.shape, generator=generator
            ),
            "source_row": torch.arange(9, dtype=torch.int64),
            "seeds": seeds,
            "indicators": ("alt", "bmi"),
            "targets": ("result_alt", "result_bmi"),
            "eye": "mean",
            "split": split,
            "transform": "train",
            "source_parquet": str(parquet),
        },
        path,
    )


class HealthDiagnosisWorkflowTests(unittest.TestCase):
    def test_fit_calibrate_and_test_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "patients.parquet"
            pd.DataFrame(
                {
                    "result_alt": [0, 0, 0, 0, 1, 0, 1, 0, -1],
                    "result_bmi": [0, 0, 0, 0, 0, 1, 1, 1, 0],
                }
            ).to_parquet(parquet)
            fit_file = root / "fit.pt"
            calibration_file = root / "calibration.pt"
            test_file = root / "test.pt"
            write_aggregate(fit_file, parquet, "train")
            write_aggregate(calibration_file, parquet, "internal_validation")
            write_aggregate(test_file, parquet, "external_validation")
            output_dir = root / "diagnosis"
            args = parse_args(
                [
                    "--fit-file",
                    str(fit_file),
                    "--calibration-file",
                    str(calibration_file),
                    "--test-file",
                    str(test_file),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            metrics = run(args)
            model_state = torch.load(
                output_dir / "health_diagnosis_model.pt", weights_only=True
            )
            test_output = torch.load(
                output_dir / "test_predictions.pt", weights_only=True
            )
            restored = HealthDiagnosticModel.from_state_dict(model_state)
            loaded = load_aggregated_predictions(test_file)

        self.assertEqual(set(metrics), {"fit", "calibration", "test"})
        self.assertEqual(metrics["test"]["labeled_patients"], 8.0)
        self.assertEqual(test_output["labels"][-1], -1)
        self.assertFalse(test_output["valid_label"][-1])
        self.assertEqual(test_output["mean_logits"].shape, (9, 2))
        self.assertEqual(
            restored.predict(loaded.logits, loaded.eye_differences).prediction.shape,
            (9,),
        )


if __name__ == "__main__":
    unittest.main()
