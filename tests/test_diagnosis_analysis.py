from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from scripts.infer.analyze_diagnosis_results import (
    SPLITS,
    load_model,
    parse_args,
    run,
    validate_alignment,
)
from src.training import select_youden_threshold


FIGURES = {
    "performance_overview.png",
    "roc_pr_curves.png",
    "confusion_matrices.png",
    "score_distributions.png",
    "score_distributions_internal_external.png",
    "calibration_curves.png",
    "health_distance.png",
    "tta_stability.png",
    "model_agreement.png",
    "feature_coefficients.png",
}


def _write_model(root: Path, name: str, *, reverse: bool = False) -> None:
    output = root / name / "diagnosis"
    output.mkdir(parents=True)
    labels = torch.tensor(([0, 1] * 15), dtype=torch.int64)
    source_rows = torch.arange(len(labels), dtype=torch.int64)
    base = torch.linspace(0.08, 0.92, len(labels))
    healthy_offset = torch.where(labels == 0, -0.16, 0.16)
    model_offset = 0.04 * torch.sin(torch.arange(len(labels), dtype=torch.float32))
    probabilities = (base * 0.45 + 0.28 + healthy_offset + model_offset).clamp(0.01, 0.99)
    if name == "retfound":
        probabilities = (probabilities * 0.88 + 0.06).roll(2)
    threshold = 0.51 if name == "resnet" else 0.46
    health_distance = 1.0 + labels.float() * 0.8 + base
    instability = 0.01 + torch.arange(len(labels), dtype=torch.float32).remainder(5) * 0.01
    order = torch.arange(len(labels) - 1, -1, -1) if reverse else torch.arange(len(labels))
    for split_index, split in enumerate(SPLITS):
        scores = (probabilities + (split_index - 1) * 0.015).clamp(0.01, 0.99)
        payload = {
            "source_row": source_rows[order],
            "labels": labels[order],
            "valid_label": torch.ones(len(labels), dtype=torch.bool)[order],
            "abnormal_probability": scores[order],
            "prediction": (scores >= threshold).to(torch.int64)[order],
            "health_distance": health_distance[order],
            "outside_fraction": (health_distance / 4.0)[order],
            "tta_instability": instability[order],
            "decision_threshold": threshold,
        }
        torch.save(payload, output / f"{split}_predictions.pt")
    torch.save(
        {
            "feature_names": ("mean_logit_alt", "mean_logit_bmi", "health_distance"),
            "classifier_weight": torch.tensor([0.4, -0.2, 0.8]),
        },
        output / "health_diagnosis_model.pt",
    )


class DiagnosisAnalysisTests(unittest.TestCase):
    def test_end_to_end_aligns_rows_and_uses_calibration_threshold(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(root, "resnet")
            _write_model(root, "retfound", reverse=True)
            output = root / "analysis"
            run(
                parse_args(
                    [
                        "--results-root",
                        str(root),
                        "--output-dir",
                        str(output),
                        "--bootstrap-samples",
                        "20",
                        "--seed",
                        "17",
                        "--dpi",
                        "60",
                    ]
                )
            )

            summary = json.loads((output / "analysis_summary.json").read_text())
            first = load_model(root, "resnet").splits["calibration"]
            second = load_model(root, "retfound").splits["calibration"]
            expected = select_youden_threshold(
                first.labels, np.mean((first.probabilities, second.probabilities), axis=0)
            )
            self.assertAlmostEqual(summary["ensemble_threshold"], expected)
            self.assertEqual(summary["bootstrap_samples"], 20)
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual({row["split"] for row in rows}, set(SPLITS))
            self.assertEqual(
                {row["model"] for row in rows}, {"resnet", "retfound", "ensemble"}
            )
            self.assertTrue((output / "bootstrap_comparisons.csv").is_file())
            self.assertIn("平均集成", (output / "analysis_report.md").read_text())
            self.assertEqual({path.name for path in (output / "figures").iterdir()}, FIGURES)
            for figure in FIGURES:
                with Image.open(output / "figures" / figure) as image:
                    image.verify()

    def test_mismatched_patient_set_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(root, "resnet")
            _write_model(root, "retfound")
            path = root / "retfound" / "diagnosis" / "test_predictions.pt"
            payload = torch.load(path, weights_only=True)
            payload["source_row"][0] = 1000
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "source_row set differs for test"):
                validate_alignment(
                    [load_model(root, "resnet"), load_model(root, "retfound")]
                )

    def test_mismatched_labels_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(root, "resnet")
            _write_model(root, "retfound")
            path = root / "retfound" / "diagnosis" / "fit_predictions.pt"
            payload = torch.load(path, weights_only=True)
            payload["labels"][0] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "labels differ for fit"):
                validate_alignment(
                    [load_model(root, "resnet"), load_model(root, "retfound")]
                )


if __name__ == "__main__":
    unittest.main()
