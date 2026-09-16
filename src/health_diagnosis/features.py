"""Features describing abnormality and test-time augmentation stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class HealthReference:
    normalization_mean: torch.Tensor
    normalization_scale: torch.Tensor
    location: torch.Tensor
    precision: torch.Tensor
    mean_distance_threshold: float
    seed_distance_threshold: float
    healthy_quantile: float


@dataclass(frozen=True)
class DiagnosticFeatures:
    values: torch.Tensor
    names: tuple[str, ...]
    mean_logits: torch.Tensor
    tta_standard_deviation: torch.Tensor
    health_distance: torch.Tensor
    outside_fraction: torch.Tensor
    mean_eye_difference: torch.Tensor


def _validate_prediction_tensors(
    logits: torch.Tensor, eye_differences: torch.Tensor | None = None
) -> tuple[int, int, int]:
    if logits.ndim != 3:
        raise ValueError("X must have shape [seeds, patients, indicators]")
    seed_count, patient_count, indicator_count = logits.shape
    if seed_count < 1 or patient_count < 1 or indicator_count < 1:
        raise ValueError("X dimensions must all be non-empty")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("X contains non-finite logits")
    if eye_differences is not None:
        if eye_differences.shape != logits.shape:
            raise ValueError("left/right differences must have the same shape as X")
        if not bool(torch.isfinite(eye_differences).all()):
            raise ValueError("left/right differences contain non-finite values")
    return seed_count, patient_count, indicator_count


def mahalanobis_distance(
    values: torch.Tensor, reference: HealthReference
) -> torch.Tensor:
    normalized = (
        values.to(dtype=torch.float32) - reference.normalization_mean
    ) / reference.normalization_scale
    centered = normalized - reference.location
    squared = torch.einsum("...i,ij,...j->...", centered, reference.precision, centered)
    return squared.clamp_min(0.0).sqrt()


def fit_health_reference(
    logits: torch.Tensor,
    abnormal_labels: torch.Tensor,
    *,
    healthy_quantile: float = 0.95,
) -> HealthReference:
    """Fit a shrinkage-covariance healthy region from patient-level labels."""
    _, patient_count, _ = _validate_prediction_tensors(logits)
    labels = abnormal_labels.to(dtype=torch.int64, device="cpu")
    if labels.shape != (patient_count,):
        raise ValueError("abnormal_labels must have shape [patients]")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("abnormal_labels must contain only 0 and 1")
    if not 0.0 < healthy_quantile < 1.0:
        raise ValueError("healthy_quantile must be between 0 and 1")

    patient_means = logits.to(dtype=torch.float32, device="cpu").mean(dim=0)
    healthy_mask = labels == 0
    healthy_means = patient_means[healthy_mask]
    if healthy_means.shape[0] < 2:
        raise ValueError("at least two healthy patients are required")

    normalization_mean = healthy_means.mean(dim=0)
    normalization_scale = healthy_means.std(dim=0, unbiased=False)
    normalization_scale = torch.where(
        normalization_scale > 1e-6,
        normalization_scale,
        torch.ones_like(normalization_scale),
    )
    normalized_healthy = (
        healthy_means - normalization_mean
    ) / normalization_scale
    estimator = LedoitWolf().fit(normalized_healthy.numpy())
    reference = HealthReference(
        normalization_mean=normalization_mean,
        normalization_scale=normalization_scale,
        location=torch.from_numpy(estimator.location_).to(dtype=torch.float32),
        precision=torch.from_numpy(estimator.precision_).to(dtype=torch.float32),
        mean_distance_threshold=0.0,
        seed_distance_threshold=0.0,
        healthy_quantile=float(healthy_quantile),
    )
    mean_distances = mahalanobis_distance(healthy_means, reference)
    healthy_seed_logits = logits[:, healthy_mask, :].to(dtype=torch.float32, device="cpu")
    seed_distances = mahalanobis_distance(healthy_seed_logits, reference)
    return HealthReference(
        normalization_mean=reference.normalization_mean,
        normalization_scale=reference.normalization_scale,
        location=reference.location,
        precision=reference.precision,
        mean_distance_threshold=float(torch.quantile(mean_distances, healthy_quantile)),
        seed_distance_threshold=float(torch.quantile(seed_distances, healthy_quantile)),
        healthy_quantile=float(healthy_quantile),
    )


def build_diagnostic_features(
    logits: torch.Tensor,
    eye_differences: torch.Tensor,
    indicators: Sequence[str],
    reference: HealthReference,
) -> DiagnosticFeatures:
    """Build mean, stability, healthy-deviation, and eye-asymmetry features."""
    _, _, indicator_count = _validate_prediction_tensors(logits, eye_differences)
    indicator_names = tuple(indicators)
    if len(indicator_names) != indicator_count:
        raise ValueError("indicator names do not match the final dimension of X")

    logits = logits.to(dtype=torch.float32, device="cpu")
    eye_differences = eye_differences.to(dtype=torch.float32, device="cpu")
    mean_logits = logits.mean(dim=0)
    tta_std = logits.std(dim=0, unbiased=False)
    health_distance = mahalanobis_distance(mean_logits, reference)
    seed_distances = mahalanobis_distance(logits, reference)
    outside_fraction = (
        seed_distances > reference.seed_distance_threshold
    ).to(dtype=torch.float32).mean(dim=0)
    mean_eye_difference = eye_differences.mean(dim=0)
    log_tta_std = (tta_std + 1e-6).log()
    values = torch.cat(
        (
            mean_logits,
            log_tta_std,
            health_distance[:, None],
            outside_fraction[:, None],
            mean_eye_difference,
        ),
        dim=1,
    )
    names = (
        tuple(f"mean_logit:{name}" for name in indicator_names)
        + tuple(f"log_tta_std:{name}" for name in indicator_names)
        + ("health_distance", "outside_fraction")
        + tuple(f"mean_abs_eye_difference:{name}" for name in indicator_names)
    )
    return DiagnosticFeatures(
        values=values,
        names=names,
        mean_logits=mean_logits,
        tta_standard_deviation=tta_std,
        health_distance=health_distance,
        outside_fraction=outside_fraction,
        mean_eye_difference=mean_eye_difference,
    )
