from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from scripts.infer.run_health_pipeline import PipelineConfig, run_pipeline


def write_prediction_root(
    root: Path,
    *,
    split: str,
    parquet: Path,
) -> None:
    rows = torch.arange(8, dtype=torch.int64)
    for indicator_index, indicator in enumerate(("alt", "bmi")):
        directory = root / indicator
        directory.mkdir(parents=True, exist_ok=True)
        for seed in (1, 2):
            values = torch.tensor(
                [0.0, 0.1, -0.1, 0.2, 3.0, 3.2, 2.8, 3.1],
                dtype=torch.float32,
            )
            values = values + indicator_index * 0.3 + (seed - 1) * 0.05
            logits = torch.stack((values, values + 0.2), dim=1)
            torch.save(
                {
                    "source_row": rows,
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


class HealthPipelineTests(unittest.TestCase):
    def test_one_command_pipeline_uses_training_pca_for_all_splits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "patients.parquet"
            pd.DataFrame(
                {
                    "result_alt": [0, 0, 0, 0, 1, 0, 1, 0],
                    "result_bmi": [0, 0, 0, 0, 0, 1, 0, 1],
                }
            ).to_parquet(parquet)
            train_root = root / "train"
            internal_root = root / "internal"
            external_root = root / "external"
            write_prediction_root(train_root, split="train", parquet=parquet)
            write_prediction_root(
                internal_root,
                split="internal_validation",
                parquet=parquet,
            )
            write_prediction_root(
                external_root,
                split="external_validation",
                parquet=parquet,
            )
            output_dir = root / "output"

            run_pipeline(
                PipelineConfig(
                    fit_predictions_dir=train_root,
                    internal_predictions_dir=internal_root,
                    external_predictions_dir=external_root,
                    output_dir=output_dir,
                    indicators=("alt", "bmi"),
                    first_seed=1,
                    last_seed=2,
                    pca_components=1,
                )
            )
            train = torch.load(
                output_dir / "aggregated" / "train.pt", weights_only=True
            )
            internal = torch.load(
                output_dir / "aggregated" / "internal_validation.pt",
                weights_only=True,
            )
            external = torch.load(
                output_dir / "aggregated" / "external_validation.pt",
                weights_only=True,
            )
            model_exists = (
                output_dir / "diagnosis" / "health_diagnosis_model.pt"
            ).is_file()
            test_predictions_exist = (
                output_dir / "diagnosis" / "test_predictions.pt"
            ).is_file()

        self.assertTrue(
            torch.equal(
                train["reducer"]["components"],
                internal["reducer"]["components"],
            )
        )
        self.assertTrue(
            torch.equal(
                train["reducer"]["components"],
                external["reducer"]["components"],
            )
        )
        self.assertTrue(model_exists)
        self.assertTrue(test_predictions_exist)


if __name__ == "__main__":
    unittest.main()
