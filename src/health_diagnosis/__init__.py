"""Patient-level health diagnosis from repeated multi-indicator logits."""

from .features import (
    DiagnosticFeatures,
    HealthReference,
    build_diagnostic_features,
    fit_health_reference,
    mahalanobis_distance,
)
from .io import (
    AggregatedPredictions,
    check_compatible,
    load_abnormal_labels,
    load_aggregated_predictions,
)
from .model import (
    DiagnosticOutput,
    HealthDiagnosticModel,
    calibrate_threshold,
    fit_health_diagnostic,
)

__all__ = [
    "AggregatedPredictions",
    "DiagnosticFeatures",
    "DiagnosticOutput",
    "HealthDiagnosticModel",
    "HealthReference",
    "build_diagnostic_features",
    "calibrate_threshold",
    "check_compatible",
    "fit_health_diagnostic",
    "fit_health_reference",
    "load_abnormal_labels",
    "load_aggregated_predictions",
    "mahalanobis_distance",
]
