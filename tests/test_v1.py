from __future__ import annotations

import numpy as np
import torch
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.model import FundusClassifier
from src.training import evaluate_paired_eyes
from src.training.metrics import binary_metrics
from train import resolve_amp_dtype


class _Pairs(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        left_logit, right_logit, target = ((2.0, -2.0, 1), (-2.0, -2.0, 0))[index]
        left = torch.tensor([left_logit]).view(1, 1, 1)
        right = torch.tensor([right_logit]).view(1, 1, 1)
        return left, right, 50, 0, target, f"p{index}"


class _FirstPixelModel(nn.Module):
    def forward(self, image: torch.Tensor, age: torch.Tensor, sex: torch.Tensor):
        return image[:, 0, 0, 0]


class V1Tests(unittest.TestCase):
    def test_auto_amp_defaults_to_float16(self) -> None:
        self.assertIs(
            resolve_amp_dtype("auto", torch.device("cuda")), torch.float16
        )

    def test_only_last_two_backbone_stages_are_trainable(self) -> None:
        model = FundusClassifier(model_name="convnext_atto", pretrained=False)
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.backbone.stem.parameters())
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for stage in model.backbone.stages[:-2]
                for parameter in stage.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for stage in model.backbone.stages[-2:]
                for parameter in stage.parameters()
            )
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.meta_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.classifier.parameters())
        )

    def test_binary_metrics_and_youden_threshold(self) -> None:
        values = binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
        )
        self.assertAlmostEqual(values["auroc"], 1.0)
        self.assertAlmostEqual(values["auprc"], 1.0)
        self.assertAlmostEqual(values["sensitivity"], 1.0)
        self.assertAlmostEqual(values["specificity"], 1.0)

    def test_evaluation_averages_two_eye_probabilities(self) -> None:
        metrics, predictions = evaluate_paired_eyes(
            _FirstPixelModel(),
            DataLoader(_Pairs(), batch_size=2),
            nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            amp_dtype=None,
            threshold=0.5,
        )
        sigmoid = torch.sigmoid(torch.tensor([2.0, -2.0])).numpy()
        self.assertAlmostEqual(predictions["left_probability"][0], sigmoid[0])
        self.assertAlmostEqual(predictions["right_probability"][0], sigmoid[1])
        self.assertAlmostEqual(predictions["probability"][0], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 1.0)

    def test_config_rejects_unknown_keys(self) -> None:
        with TemporaryDirectory() as directory:
            override = Path(directory) / "bad.yaml"
            override.write_text("training:\n  typo: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training.typo"):
                load_config(override)


if __name__ == "__main__":
    unittest.main()
