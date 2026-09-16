#!/usr/bin/env python3
"""One-command aggregation, shared PCA, and health-diagnosis workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


# ======================== 只需要修改下面四个路径 ========================
FIT_PREDICTIONS_DIR: Path | None = Path("/DaTa/ljc_data/infers/retfound_dali_balanced_tr")          # 训练集的 alt/bmi/... 根目录
INTERNAL_PREDICTIONS_DIR: Path | None = Path("/DaTa/ljc_data/infers/retfound_dali_balanced_iv")     # 内部验证集的 alt/bmi/... 根目录
EXTERNAL_PREDICTIONS_DIR: Path | None = Path("/DaTa/ljc_data/infers/retfound_dali_balanced_ev")     # 外部验证集的 alt/bmi/... 根目录
OUTPUT_DIR: Path | None = Path("/DaTa/ljc_data/preds/retfound_dali_balanced")                   # 聚合结果、模型和评估结果目录
# ========================================================================


# 一般不需要修改以下设置。
INDICATORS = (
    "alt",
    "bmi",
    "fbg",
    "hb",
    "hba1c",
    "hct",
    "hdl_c",
    "rbc",
    "scr",
    "tg",
    "wbc",
)
FIRST_SEED = 2026
LAST_SEED = 2075
EYE = "mean"
PCA_COMPONENTS = 3
PCA_FIT = "joint"
HEALTHY_QUANTILE = 0.95
LOGISTIC_C = 1.0
PARQUET_OVERRIDE: Path | None = None


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.infer.fit_health_diagnosis import (  # noqa: E402
    parse_args as parse_diagnosis_args,
)
from scripts.infer.fit_health_diagnosis import run as run_diagnosis  # noqa: E402
from scripts.infer.reduce_predictions import (  # noqa: E402
    parse_args as parse_reduction_args,
)
from scripts.infer.reduce_predictions import run as run_reduction  # noqa: E402


@dataclass(frozen=True)
class PipelineConfig:
    fit_predictions_dir: Path
    internal_predictions_dir: Path
    external_predictions_dir: Path
    output_dir: Path
    indicators: tuple[str, ...] = INDICATORS
    first_seed: int = FIRST_SEED
    last_seed: int = LAST_SEED
    eye: str = EYE
    pca_components: int = PCA_COMPONENTS
    pca_fit: str = PCA_FIT
    healthy_quantile: float = HEALTHY_QUANTILE
    logistic_c: float = LOGISTIC_C
    parquet_override: Path | None = PARQUET_OVERRIDE


def _reduction_arguments(
    config: PipelineConfig,
    *,
    input_dir: Path,
    output_file: Path,
    split: str,
    pca_reference: Path | None = None,
) -> list[str]:
    arguments = [
        "--input-dir",
        str(input_dir),
        "--output-file",
        str(output_file),
        "--split",
        split,
        "--indicators",
        *config.indicators,
        "--seeds",
        "seq",
        str(config.first_seed),
        str(config.last_seed),
        "--eye",
        config.eye,
        "--n-components",
        str(config.pca_components),
        "--pca-fit",
        config.pca_fit,
    ]
    if pca_reference is not None:
        arguments.extend(("--pca-reference", str(pca_reference)))
    return arguments


def run_pipeline(config: PipelineConfig) -> Path:
    if config.first_seed > config.last_seed:
        raise ValueError("first_seed must not exceed last_seed")
    output_dir = config.output_dir.resolve()
    aggregate_dir = output_dir / "aggregated"
    diagnosis_dir = output_dir / "diagnosis"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    fit_file = aggregate_dir / "train.pt"
    internal_file = aggregate_dir / "internal_validation.pt"
    external_file = aggregate_dir / "external_validation.pt"

    print("[1/4] 聚合训练集并拟合共享 PCA", flush=True)
    run_reduction(
        parse_reduction_args(
            _reduction_arguments(
                config,
                input_dir=config.fit_predictions_dir,
                output_file=fit_file,
                split="train",
            )
        )
    )
    print("[2/4] 聚合内部验证集并应用训练集 PCA", flush=True)
    run_reduction(
        parse_reduction_args(
            _reduction_arguments(
                config,
                input_dir=config.internal_predictions_dir,
                output_file=internal_file,
                split="internal_validation",
                pca_reference=fit_file,
            )
        )
    )
    print("[3/4] 聚合外部验证集并应用训练集 PCA", flush=True)
    run_reduction(
        parse_reduction_args(
            _reduction_arguments(
                config,
                input_dir=config.external_predictions_dir,
                output_file=external_file,
                split="external_validation",
                pca_reference=fit_file,
            )
        )
    )

    diagnosis_arguments = [
        "--fit-file",
        str(fit_file),
        "--calibration-file",
        str(internal_file),
        "--test-file",
        str(external_file),
        "--output-dir",
        str(diagnosis_dir),
        "--healthy-quantile",
        str(config.healthy_quantile),
        "--logistic-c",
        str(config.logistic_c),
    ]
    if config.parquet_override is not None:
        diagnosis_arguments.extend(("--parquet", str(config.parquet_override)))
    print("[4/4] 拟合健康诊断、内部校准并评估外部集", flush=True)
    run_diagnosis(parse_diagnosis_args(diagnosis_arguments))
    print(f"全部完成：{output_dir}", flush=True)
    return output_dir


def configured_pipeline() -> PipelineConfig:
    paths = {
        "FIT_PREDICTIONS_DIR": FIT_PREDICTIONS_DIR,
        "INTERNAL_PREDICTIONS_DIR": INTERNAL_PREDICTIONS_DIR,
        "EXTERNAL_PREDICTIONS_DIR": EXTERNAL_PREDICTIONS_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
    }
    missing = [name for name, value in paths.items() if value is None]
    if missing:
        raise ValueError(
            "请先在脚本顶部填写路径：" + ", ".join(missing)
        )
    return PipelineConfig(
        fit_predictions_dir=FIT_PREDICTIONS_DIR,
        internal_predictions_dir=INTERNAL_PREDICTIONS_DIR,
        external_predictions_dir=EXTERNAL_PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR,
    )


def main() -> int:
    run_pipeline(configured_pipeline())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
