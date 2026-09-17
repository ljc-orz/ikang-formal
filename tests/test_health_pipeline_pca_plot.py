from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from PIL import Image

from scripts.infer.plot_health_pipeline_pca import parse_args, run


def write_aggregate(
    path: Path, parquet: Path, split: str, components: torch.Tensor
) -> None:
    generator = torch.Generator().manual_seed(7)
    patient_count = 12
    seed_count = 3
    x_prime = torch.randn(
        seed_count, patient_count, 3, generator=generator
    )
    x_prime[:, 6:] += torch.tensor([2.0, 1.0, -1.0])
    torch.save(
        {
            "X_prime": x_prime,
            "source_row": torch.arange(patient_count, dtype=torch.int64),
            "seeds": torch.tensor([2026, 2027, 2028], dtype=torch.int64),
            "indicators": ("alt", "bmi", "fbg"),
            "targets": ("result_alt", "result_bmi", "result_fbg"),
            "eye": "mean",
            "split": split,
            "transform": "train",
            "source_parquet": str(parquet),
            "reducer": {
                "method": "pca",
                "fit_scope": "joint",
                "components": components,
                "explained_variance_ratio": torch.tensor([0.5, 0.3, 0.2]),
            },
        },
        path,
    )


class HealthPipelinePcaPlotTests(unittest.TestCase):
    def test_writes_one_valid_figure_per_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "patients.parquet"
            pd.DataFrame(
                {
                    "result_alt": [0] * 6 + [1] * 6,
                    "result_bmi": [0] * 12,
                    "result_fbg": [0] * 11 + [-1],
                }
            ).to_parquet(parquet)
            aggregate_dir = root / "pipeline" / "aggregated"
            aggregate_dir.mkdir(parents=True)
            components = torch.eye(3)
            for filename, split in (
                ("train.pt", "train"),
                ("internal_validation.pt", "internal_validation"),
                ("external_validation.pt", "external_validation"),
            ):
                write_aggregate(aggregate_dir / filename, parquet, split, components)
            output = run(
                parse_args(
                    [
                        "--health-pipeline-output",
                        str(root / "pipeline"),
                        "--dpi",
                        "60",
                    ]
                )
            )
            summary = json.loads((output / "pca_summary.json").read_text())
            figures = sorted(output.glob("*.png"))
            self.assertEqual(len(figures), 3)
            self.assertEqual(summary["seed_count"], 3)
            self.assertEqual(summary["datasets"]["train"]["healthy"], 6)
            self.assertEqual(summary["datasets"]["train"]["abnormal"], 5)
            self.assertEqual(summary["datasets"]["train"]["incomplete_label"], 1)
            for figure in figures:
                with Image.open(figure) as image:
                    image.verify()


if __name__ == "__main__":
    unittest.main()
