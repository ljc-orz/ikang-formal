from __future__ import annotations

import unittest

import torch

from src.health_diagnosis import (
    HealthDiagnosticModel,
    build_diagnostic_features,
    calibrate_threshold,
    fit_health_diagnostic,
    fit_health_reference,
)


class HealthDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        healthy = torch.randn(8, 12, 3, generator=generator) * 0.15
        abnormal = 3.0 + torch.randn(8, 12, 3, generator=generator) * 0.2
        self.logits = torch.cat((healthy, abnormal), dim=1)
        self.eye_differences = torch.rand(
            self.logits.shape, generator=generator
        ) * 0.1
        self.labels = torch.tensor([0] * 12 + [1] * 12, dtype=torch.int64)
        self.indicators = ("alt", "bmi", "fbg")

    def test_features_combine_mean_stability_distance_and_eye_gap(self) -> None:
        reference = fit_health_reference(self.logits, self.labels)
        features = build_diagnostic_features(
            self.logits,
            self.eye_differences,
            self.indicators,
            reference,
        )

        self.assertEqual(features.values.shape, (24, 11))
        self.assertEqual(len(features.names), 11)
        self.assertTrue(
            features.health_distance[12:].mean()
            > features.health_distance[:12].mean()
        )
        self.assertTrue(
            bool(((features.outside_fraction >= 0) & (features.outside_fraction <= 1)).all())
        )

    def test_logistic_model_round_trip_and_threshold_calibration(self) -> None:
        model = fit_health_diagnostic(
            self.logits,
            self.eye_differences,
            self.labels,
            self.indicators,
        )
        calibrated = calibrate_threshold(
            model,
            self.logits,
            self.eye_differences,
            self.labels,
        )
        restored = HealthDiagnosticModel.from_state_dict(calibrated.to_state_dict())

        output = restored.predict(self.logits, self.eye_differences)

        self.assertEqual(output.abnormal_probability.shape, (24,))
        self.assertEqual(output.mean_logits.shape, (24, 3))
        self.assertGreater(output.abnormal_probability[12:].mean(), 0.9)
        self.assertLess(output.abnormal_probability[:12].mean(), 0.1)
        self.assertGreater(calibrated.decision_threshold, 0.0)
        self.assertLess(calibrated.decision_threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
