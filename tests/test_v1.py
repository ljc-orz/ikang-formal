from __future__ import annotations

import json
import numpy as np
import torch
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import apply_config_overrides, load_config
from src.data import BalancedBinaryLoader
from src.data.dali_webdataset import DaliFundusLoader, read_dali_dataset_spec
from src.model import FundusClassifier, LoRALinear, available_backbones
from src.training import evaluate_paired_eyes
from src.training.metrics import binary_metrics
from train import make_criterion, resolve_amp_dtype


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


def _training_batch(labels: list[int]):
    values = torch.tensor(labels)
    images = torch.arange(len(labels)).view(-1, 1, 1, 1).float()
    return images, values + 40, values, values


class V1Tests(unittest.TestCase):
    def test_default_data_backend_does_not_require_dali(self) -> None:
        self.assertEqual(load_config()["data"]["backend"], "torchvision")
        self.assertEqual(load_config()["data"]["train_sampling"], "natural")

    def test_balanced_loader_keeps_the_natural_epoch_length(self) -> None:
        source = [
            _training_batch([0, 0, 0, 0]),
            _training_batch([1, 1, 0, 0]),
            _training_batch([0, 0, 1, 1]),
        ]
        loader = BalancedBinaryLoader(
            source,
            batch_size=4,
            class_counts={-1: 0, 0: 4, 1: 2},
            seed=2026,
        )
        labels = torch.cat([batch[-1] for batch in loader])
        self.assertEqual(labels.tolist().count(0), 6)
        self.assertEqual(labels.tolist().count(1), 6)
        self.assertEqual(labels.numel(), 12)
        self.assertEqual(len(loader), 3)

    def test_balanced_sampling_disables_automatic_positive_weight(self) -> None:
        config = load_config()
        config["data"]["train_sampling"] = "balanced"
        _, weight = make_criterion(config, {-1: 0, 0: 4, 1: 2}, torch.device("cpu"))
        self.assertEqual(weight, 1.0)

    def test_default_backbone_remains_convnext(self) -> None:
        self.assertEqual(load_config()["model"]["backbone"], "convnext")
        self.assertEqual(
            available_backbones(), ("convnext", "retfound_dinov2", "resnet50")
        )

    def test_resnet_only_unfreezes_requested_final_blocks(self) -> None:
        model = FundusClassifier(
            backbone_type="resnet50",
            model_name="resnet50.a1_in1k",
            pretrained=False,
            trainable_last_n_blocks=2,
        )
        blocks = [
            block
            for name in ("layer1", "layer2", "layer3", "layer4")
            for block in getattr(model.backbone, name)
        ]
        self.assertEqual(len(blocks), 16)
        self.assertFalse(
            any(parameter.requires_grad for block in blocks[:-2] for parameter in block.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for block in blocks[-2:] for parameter in block.parameters())
        )
        self.assertFalse(model.backbone.conv1.weight.requires_grad)

    def test_lora_starts_as_an_exact_no_op(self) -> None:
        torch.manual_seed(11)
        base = nn.Linear(5, 3)
        value = torch.randn(2, 5)
        expected = base(value)
        layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
        torch.testing.assert_close(layer(value), expected)
        self.assertFalse(layer.base.weight.requires_grad)

    def test_dali_manifest_can_be_validated_without_importing_dali(self) -> None:
        spec = read_dali_dataset_spec("example/webdataset", "train")
        manifest = json.loads(
            (Path("example/webdataset") / "dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(spec.sample_count, manifest["splits"]["train"])
        self.assertEqual(len(spec.tar_paths), len(spec.index_paths))
        self.assertTrue(all(Path(path).is_file() for path in spec.index_paths))

    def test_dali_training_requires_even_image_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "even --batch-size"):
            DaliFundusLoader(
                "unused",
                "train",
                "result_alt",
                mode="eyes",
                batch_size=3,
                num_threads=1,
                device_id=0,
                image_size=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                seed=2026,
                skip_missing_target=True,
            )

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

    def test_dotted_config_overrides_parse_yaml_values(self) -> None:
        config = load_config("src/config/resnet50.yaml")
        apply_config_overrides(
            config,
            [
                "training.max_epochs=12",
                "training.backbone_lr=5e-5",
                "model.pretrained=false",
                "model.trainable_last_n_blocks=5",
            ],
        )
        self.assertEqual(config["training"]["max_epochs"], 12)
        self.assertEqual(config["training"]["backbone_lr"], 5e-5)
        self.assertIsInstance(config["training"]["backbone_lr"], float)
        self.assertFalse(config["model"]["pretrained"])
        self.assertEqual(config["model"]["trainable_last_n_blocks"], 5)

    def test_dotted_config_overrides_reject_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "training.typo"):
            apply_config_overrides(load_config(), ["training.typo=1"])


if __name__ == "__main__":
    unittest.main()
