#!/usr/bin/env python3
"""统计分层 Parquet 中各 split、各 result_* 指标的异常患病率。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq


EXPECTED_SPLIT_ORDER = (
    "train",
    "internal_validation",
    "external_validation",
)
INDICATOR_PREFIX = "result_"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="stratified_hospital_split.py 结果导出的 Parquet 文件",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="可选：将 split × 指标的详细统计保存为 CSV",
    )
    return parser.parse_args(argv)


def discover_columns(path: Path) -> list[str]:
    schema_names = pq.ParquetFile(path).schema_arrow.names
    if "split" not in schema_names:
        raise ValueError("Parquet 缺少必需的 split 列")
    indicators = [name for name in schema_names if name.startswith(INDICATOR_PREFIX)]
    if not indicators:
        raise ValueError(f"Parquet 中没有 {INDICATOR_PREFIX}* 指标列")
    return indicators


def ordered_splits(values: pd.Series) -> list[str]:
    if values.isna().any():
        raise ValueError("split 列包含空值")
    observed = [str(value) for value in values.drop_duplicates().tolist()]
    if any(not value.strip() for value in observed):
        raise ValueError("split 列包含空字符串")
    expected = [name for name in EXPECTED_SPLIT_ORDER if name in observed]
    extras = sorted(set(observed) - set(EXPECTED_SPLIT_ORDER))
    return expected + extras


def validate_labels(frame: pd.DataFrame, indicators: Sequence[str]) -> None:
    errors: list[str] = []
    for indicator in indicators:
        values = frame[indicator]
        invalid = values.notna() & ~values.isin((0, 1))
        if invalid.any():
            examples = values.loc[invalid].head(5).tolist()
            errors.append(f"{indicator}={examples}")
    if errors:
        raise ValueError(
            "result_* 标签只能是 0、1 或空值；发现非法值：" + ", ".join(errors)
        )


def summarize_parquet(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    source = path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"不是文件：{source}")
    indicators = discover_columns(source)
    frame = pd.read_parquet(source, columns=["split", *indicators])
    splits = ordered_splits(frame["split"])
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str)
    validate_labels(frame, indicators)

    total_patients = len(frame)
    split_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for split in splits:
        selected = frame.loc[frame["split"] == split]
        split_patients = len(selected)
        split_rows.append(
            {
                "split": split,
                "patients": split_patients,
                "dataset_fraction": (
                    split_patients / total_patients if total_patients else None
                ),
            }
        )
        for indicator in indicators:
            values = selected[indicator]
            labeled = int(values.notna().sum())
            abnormal = int((values == 1).sum())
            normal = int((values == 0).sum())
            missing = split_patients - labeled
            detail_rows.append(
                {
                    "split": split,
                    "split_patients": split_patients,
                    "indicator": indicator,
                    "labeled_patients": labeled,
                    "missing_patients": missing,
                    "normal_patients": normal,
                    "abnormal_patients": abnormal,
                    "abnormal_prevalence": abnormal / labeled if labeled else None,
                }
            )
    return split_rows, detail_rows, indicators


def percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def print_summary(
    source: Path,
    split_rows: Sequence[dict[str, Any]],
    detail_rows: Sequence[dict[str, Any]],
    indicator_count: int,
) -> None:
    total = sum(int(row["patients"]) for row in split_rows)
    print(f"Parquet: {source.expanduser().resolve()}")
    print(f"总人数: {total:,}；指标数: {indicator_count}")
    print()
    print(f"{'split':<22} {'人数':>10} {'占全部数据':>12}")
    print("-" * 48)
    for row in split_rows:
        print(
            f"{row['split']:<22} {row['patients']:>10,} "
            f"{percentage(row['dataset_fraction']):>12}"
        )

    print()
    print(
        f"{'split':<22} {'指标':<20} {'有效':>8} {'缺失':>8} "
        f"{'正常':>8} {'异常':>8} {'异常患病率':>12}"
    )
    print("-" * 96)
    for row in detail_rows:
        print(
            f"{row['split']:<22} {row['indicator']:<20} "
            f"{row['labeled_patients']:>8,} {row['missing_patients']:>8,} "
            f"{row['normal_patients']:>8,} {row['abnormal_patients']:>8,} "
            f"{percentage(row['abnormal_prevalence']):>12}"
        )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "split_patients",
        "indicator",
        "labeled_patients",
        "missing_patients",
        "normal_patients",
        "abnormal_patients",
        "abnormal_prevalence",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def run(args: argparse.Namespace) -> None:
    split_rows, detail_rows, indicators = summarize_parquet(args.parquet)
    print_summary(args.parquet, split_rows, detail_rows, len(indicators))
    if args.output_csv is not None:
        destination = write_csv(args.output_csv, detail_rows)
        print(f"\nCSV 已保存：{destination}")


def main(argv: Sequence[str] | None = None) -> int:
    parser_args = parse_args(argv)
    try:
        run(parser_args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
