from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from src.heatmap_vis import generate_heatmaps, generate_heatmaps_batch
from src.heatmap_vis.core import reshape_patch_tokens


class TinyResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())

    def forward(self, image):
        return self.layer4(image).mean(dim=(-2, -1))


class TinyFundusModel(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "resnet50":
            self.backbone = TinyResNetBackbone()
            dimension = 8
        else:
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=48,
                depth=3,
                num_heads=3,
                mlp_ratio=2,
                num_classes=0,
            )
            dimension = 48
        self.meta_encoder = nn.Linear(2, 4)
        self.classifier = nn.Linear(dimension + 4, 1)

    def forward(self, image, age, sex):
        image_feature = self.backbone(image)
        metadata = torch.stack((age.float() / 100.0, sex.float()), dim=1)
        return self.classifier(
            torch.cat((image_feature, self.meta_encoder(metadata)), dim=1)
        ).squeeze(1)


class HeatmapVisTests(unittest.TestCase):
    def assert_heatmap(self, value):
        self.assertEqual(tuple(value.shape), (32, 32))
        self.assertTrue(torch.isfinite(value).all())
        self.assertGreaterEqual(float(value.min()), 0.0)
        self.assertLessEqual(float(value.max()), 1.0)

    def inputs(self):
        return (
            torch.randn(1, 3, 32, 32),
            torch.tensor([52.0]),
            torch.tensor([0.0]),
        )

    def test_actual_forward_signature_resnet_gradcam(self):
        model = TinyFundusModel("resnet50").eval()
        result = generate_heatmaps(model, *self.inputs())
        self.assertEqual(set(result.heatmaps), {"resnet_gradcam"})
        self.assert_heatmap(result.heatmaps["resnet_gradcam"])
        self.assertAlmostEqual(result.probability, torch.sigmoid(torch.tensor(result.logit)).item())

    def test_actual_forward_signature_dinov2_methods(self):
        model = TinyFundusModel("retfound_dinov2").eval()
        result = generate_heatmaps(model, *self.inputs(), explanation_target="abnormal")
        self.assertEqual(
            set(result.heatmaps),
            {
                "dinov2_gradcam",
                "dinov2_gradient_rollout",
                "dinov2_cls_attention",
            },
        )
        for heatmap in result.heatmaps.values():
            self.assert_heatmap(heatmap)

    def test_batch_native_resolution_keeps_eye_axis_and_patch_grid(self):
        model = TinyFundusModel("retfound_dinov2").eval()
        image, age, sex = self.inputs()
        result = generate_heatmaps_batch(
            model,
            image.repeat(2, 1, 1, 1),
            age.repeat(2),
            sex.repeat(2),
            native_resolution=True,
        )
        self.assertEqual(tuple(result.logits.shape), (2,))
        for heatmap in result.heatmaps.values():
            self.assertEqual(tuple(heatmap.shape), (2, 4, 4))
            self.assertTrue(torch.isfinite(heatmap).all())

    def test_gradient_capture_does_not_register_tensor_hooks(self):
        model = TinyFundusModel("retfound_dinov2").eval()
        with patch.object(
            torch.Tensor,
            "register_hook",
            side_effect=AssertionError("tensor hook would retain the computation graph"),
        ):
            result = generate_heatmaps(model, *self.inputs())
        self.assertEqual(len(result.heatmaps), 3)

    def test_fp16_amp_option_falls_back_cleanly_on_cpu(self):
        model = TinyFundusModel("retfound_dinov2").eval()
        result = generate_heatmaps(
            model,
            *self.inputs(),
            amp_dtype=torch.float16,
        )
        for heatmap in result.heatmaps.values():
            self.assert_heatmap(heatmap)

    def test_predicted_healthy_uses_negative_logit(self):
        model = TinyFundusModel("resnet50").eval()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(-2.0)
        result = generate_heatmaps(model, *self.inputs(), explanation_target="predicted")
        self.assertEqual(result.predicted_label, 0)
        self.assertAlmostEqual(result.objective, 2.0, places=6)

    def test_prefix_token_is_removed_before_grid_reshape(self):
        tokens = torch.arange(17 * 8).reshape(1, 17, 8).float()
        result = reshape_patch_tokens(tokens, (4, 4), 1)
        self.assertEqual(tuple(result.shape), (1, 8, 4, 4))
        self.assertEqual(float(result[0, 0, 0, 0]), float(tokens[0, 1, 0]))


if __name__ == "__main__":
    unittest.main()
