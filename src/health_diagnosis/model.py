"""Logistic patient-level diagnosis using healthy-region and stability features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training.metrics import select_youden_threshold

from .features import (
    DiagnosticFeatures,
    HealthReference,
    build_diagnostic_features,
    fit_health_reference,
)


@dataclass(frozen=True)
class DiagnosticOutput:
    abnormal_probability: torch.Tensor
    prediction: torch.Tensor
    health_distance: torch.Tensor
    outside_fraction: torch.Tensor
    tta_instability: torch.Tensor
    mean_logits: torch.Tensor
    tta_standard_deviation: torch.Tensor
    mean_eye_difference: torch.Tensor


@dataclass(frozen=True)
class HealthDiagnosticModel:
    indicators: tuple[str, ...]
    feature_names: tuple[str, ...]
    health_reference: HealthReference
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    classifier_weight: torch.Tensor
    classifier_bias: float
    decision_threshold: float

    def diagnostic_features(
        self, logits: torch.Tensor, eye_differences: torch.Tensor
    ) -> DiagnosticFeatures:
        return build_diagnostic_features(
            logits,
            eye_differences,
            self.indicators,
            self.health_reference,
        )

    def predict(
        self, logits: torch.Tensor, eye_differences: torch.Tensor
    ) -> DiagnosticOutput:
        features = self.diagnostic_features(logits, eye_differences)
        normalized = (features.values - self.feature_mean) / self.feature_scale
        classifier_logits = (
            normalized @ self.classifier_weight + self.classifier_bias
        )
        probabilities = classifier_logits.sigmoid()
        return DiagnosticOutput(
            abnormal_probability=probabilities,
            prediction=(probabilities >= self.decision_threshold).to(torch.int64),
            health_distance=features.health_distance,
            outside_fraction=features.outside_fraction,
            tta_instability=features.tta_standard_deviation.square().sum(dim=1),
            mean_logits=features.mean_logits,
            tta_standard_deviation=features.tta_standard_deviation,
            mean_eye_difference=features.mean_eye_difference,
        )

    def to_state_dict(self) -> dict[str, Any]:
        reference = self.health_reference
        return {
            "format_version": 1,
            "indicators": self.indicators,
            "feature_names": self.feature_names,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "classifier_weight": self.classifier_weight,
            "classifier_bias": self.classifier_bias,
            "decision_threshold": self.decision_threshold,
            "health_reference": {
                "normalization_mean": reference.normalization_mean,
                "normalization_scale": reference.normalization_scale,
                "location": reference.location,
                "precision": reference.precision,
                "mean_distance_threshold": reference.mean_distance_threshold,
                "seed_distance_threshold": reference.seed_distance_threshold,
                "healthy_quantile": reference.healthy_quantile,
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> HealthDiagnosticModel:
        reference = state["health_reference"]
        return cls(
            indicators=tuple(state["indicators"]),
            feature_names=tuple(state["feature_names"]),
            health_reference=HealthReference(
                normalization_mean=reference["normalization_mean"],
                normalization_scale=reference["normalization_scale"],
                location=reference["location"],
                precision=reference["precision"],
                mean_distance_threshold=float(reference["mean_distance_threshold"]),
                seed_distance_threshold=float(reference["seed_distance_threshold"]),
                healthy_quantile=float(reference["healthy_quantile"]),
            ),
            feature_mean=state["feature_mean"],
            feature_scale=state["feature_scale"],
            classifier_weight=state["classifier_weight"],
            classifier_bias=float(state["classifier_bias"]),
            decision_threshold=float(state["decision_threshold"]),
        )


def fit_health_diagnostic(
    logits: torch.Tensor,
    eye_differences: torch.Tensor,
    abnormal_labels: torch.Tensor,
    indicators: Sequence[str],
    *,
    healthy_quantile: float = 0.95,
    logistic_c: float = 1.0,
) -> HealthDiagnosticModel:
    labels = abnormal_labels.to(dtype=torch.int64, device="cpu")
    if len(torch.unique(labels)) != 2:
        raise ValueError("diagnostic fitting requires both healthy and abnormal patients")
    if logistic_c <= 0:
        raise ValueError("logistic_c must be positive")

    reference = fit_health_reference(
        logits,
        labels,
        healthy_quantile=healthy_quantile,
    )
    features = build_diagnostic_features(
        logits, eye_differences, indicators, reference
    )
    scaler = StandardScaler().fit(features.values.numpy())
    normalized = scaler.transform(features.values.numpy())
    classifier = LogisticRegression(
        C=logistic_c,
        class_weight="balanced",
        max_iter=2_000,
        solver="lbfgs",
        random_state=0,
    ).fit(normalized, labels.numpy())
    return HealthDiagnosticModel(
        indicators=tuple(indicators),
        feature_names=features.names,
        health_reference=reference,
        feature_mean=torch.from_numpy(scaler.mean_).to(dtype=torch.float32),
        feature_scale=torch.from_numpy(scaler.scale_).to(dtype=torch.float32),
        classifier_weight=torch.from_numpy(classifier.coef_[0]).to(dtype=torch.float32),
        classifier_bias=float(classifier.intercept_[0]),
        decision_threshold=0.5,
    )


def calibrate_threshold(
    model: HealthDiagnosticModel,
    logits: torch.Tensor,
    eye_differences: torch.Tensor,
    abnormal_labels: torch.Tensor,
) -> HealthDiagnosticModel:
    label_tensor = abnormal_labels.to(dtype=torch.int64, device="cpu")
    if len(torch.unique(label_tensor)) != 2:
        raise ValueError("threshold calibration requires both healthy and abnormal patients")
    probabilities = model.predict(logits, eye_differences).abnormal_probability.numpy()
    labels = label_tensor.numpy()
    threshold = select_youden_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )
    return replace(model, decision_threshold=float(threshold))
