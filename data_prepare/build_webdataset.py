#!/usr/bin/env python3
"""Convert paired fundus images described by a Parquet file to WebDataset shards.

Each output sample has four components::

    00000042.left.jpg
    00000042.right.jpg
    00000042.labels.npy
    00000042.meta.json

The tar archives are uncompressed so NVIDIA DALI can mmap them and use the
sidecar indexes produced by ``wds2idx``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import webdataset as wds


SPLITS = ("train", "external_validation", "internal_validation")
IMAGE_COLUMNS = ("image_path_1", "image_path_2")
LABEL_PREFIX = "result_"
LABEL_MISSING_VALUE = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sharded WebDataset tar files and NVIDIA DALI indexes."
    )
    parser.add_argument("--parquet", type=Path, required=True, help="Input Parquet file")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory relative to which image_path_1 and image_path_2 are resolved",
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=2_000,
        help="Maximum patients in one tar (default: 2000)",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=2.0,
        help="Approximate maximum component bytes in one tar (default: 2.0 GiB)",
    )
    parser.add_argument(
        "--missing-image",
        choices=("error", "skip"),
        default="error",
        help="Fail on a missing image, or skip its whole patient row (default: error)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory, only after a new dataset is complete",
    )
    parser.add_argument(
        "--no-dali-index",
        action="store_true",
        help="Do not create .idx sidecars (DALI can infer them, but starts more slowly)",
    )
    return parser.parse_args()


def json_value(value: Any) -> Any:
    """Convert pandas/NumPy scalars to strict JSON values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def npy_bytes(array: np.ndarray) -> bytes:
    """Serialize an ndarray using the NPY v1.0 format supported by DALI."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def resolve_image(image_root: Path, value: Any, row_number: int, column: str) -> Path:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"row {row_number}: {column} is empty")
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"row {row_number}: {column} must be relative, got {value!r}")
    candidate = (image_root / relative).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: {column} escapes --image-root: {value!r}"
        ) from exc
    return candidate


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    required = {"split", "examage", "usex", *IMAGE_COLUMNS}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Parquet is missing required columns: {missing_columns}")

    label_columns = [str(column) for column in df.columns if str(column).startswith(LABEL_PREFIX)]
    if not label_columns:
        raise ValueError(f"Parquet has no {LABEL_PREFIX}* columns")

    observed_splits = set(df["split"].dropna().astype(str).unique())
    unknown_splits = sorted(observed_splits - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown split values: {unknown_splits}; expected {list(SPLITS)}")
    missing_splits = sorted(set(SPLITS) - observed_splits)
    if missing_splits:
        raise ValueError(f"Parquet contains no rows for split(s): {missing_splits}")
    if df["split"].isna().any():
        raise ValueError("split contains null values")

    invalid_labels: list[str] = []
    for column in label_columns:
        valid = df[column].isna() | df[column].isin((0, 1))
        if not bool(valid.all()):
            examples = df.loc[~valid, column].head(5).tolist()
            invalid_labels.append(f"{column}={examples}")
    if invalid_labels:
        raise ValueError(
            "result_* values must be 0, 1, or null; invalid values: "
            + ", ".join(invalid_labels)
        )
    return label_columns


def preflight_images(
    df: pd.DataFrame, image_root: Path, missing_policy: str
) -> tuple[dict[int, tuple[Path, Path]], list[dict[str, Any]]]:
    """Resolve every input path before creating output archives."""
    resolved: dict[int, tuple[Path, Path]] = {}
    skipped: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(df.iterrows()):
        paths = tuple(
            resolve_image(image_root, row[column], row_number, column)
            for column in IMAGE_COLUMNS
        )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            detail = {
                "source_row": row_number,
                "uid": json_value(row.get("uid")),
                "id": json_value(row.get("id")),
                "missing_paths": missing,
            }
            if missing_policy == "error":
                raise FileNotFoundError(
                    f"row {row_number} is missing image(s): {missing}; "
                    "use --missing-image skip to omit and audit it"
                )
            skipped.append(detail)
            continue
        resolved[row_number] = (paths[0], paths[1])
    return resolved, skipped


def encode_labels(row: pd.Series, label_columns: list[str]) -> np.ndarray:
    values = [
        LABEL_MISSING_VALUE if pd.isna(row[column]) else int(row[column])
        for column in label_columns
    ]
    return np.asarray(values, dtype=np.int8)


def encode_metadata(
    row: pd.Series, row_number: int, metadata_columns: Iterable[str]
) -> bytes:
    metadata = {column: json_value(row[column]) for column in metadata_columns}
    metadata["source_row"] = row_number
    return json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def build_archives(
    df: pd.DataFrame,
    image_paths: dict[int, tuple[Path, Path]],
    output: Path,
    label_columns: list[str],
    max_count: int,
    max_size: int,
) -> Counter[str]:
    metadata_columns = [column for column in df.columns if column not in label_columns]
    counts: Counter[str] = Counter()

    for split in SPLITS:
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        selected = [
            row_number
            for row_number in image_paths
            if str(df.iloc[row_number]["split"]) == split
        ]
        if not selected:
            continue

        pattern = str(split_dir / f"{split}-%06d.tar")
        with wds.ShardWriter(
            pattern,
            maxcount=max_count,
            maxsize=max_size,
            encoder=False,
            mtime=0,
            user="",
            group="",
        ) as sink:
            for row_number in selected:
                row = df.iloc[row_number]
                left_path, right_path = image_paths[row_number]
                # A numeric source-row key is unique, stable for this Parquet, and
                # contains no dot (DALI treats text after the first dot as extension).
                key = f"{row_number:08d}"
                sink.write(
                    {
                        "__key__": key,
                        "left.jpg": left_path.read_bytes(),
                        "right.jpg": right_path.read_bytes(),
                        "labels.npy": npy_bytes(encode_labels(row, label_columns)),
                        "meta.json": encode_metadata(row, row_number, metadata_columns),
                    }
                )
                counts[split] += 1
    return counts


def create_dali_indexes(output: Path) -> None:
    executable = shutil.which("wds2idx")
    if executable is None:
        raise RuntimeError(
            "wds2idx was not found on PATH; run this script inside the NVIDIA DALI "
            "environment, or pass --no-dali-index"
        )
    # wds2idx parses the human-readable output of the system `tar` command and
    # expects its keywords/date fields in English.  Keep the caller's locale for
    # the rest of the build, but make this fragile subprocess deterministic.
    index_env = os.environ.copy()
    index_env.update({"LANG": "C", "LC_ALL": "C"})
    for tar_path in sorted(output.glob("*/*.tar")):
        index_path = tar_path.with_suffix(".idx")
        try:
            subprocess.run(
                [executable, str(tar_path), str(index_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=index_env,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stdout or "").strip() or "wds2idx produced no output"
            raise RuntimeError(
                f"wds2idx failed for {tar_path} with exit code {exc.returncode}:\n"
                f"{detail}"
            ) from exc


def shard_records(output: Path, with_indexes: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tar_path in sorted(output.glob("*/*.tar")):
        relative_tar = tar_path.relative_to(output)
        record: dict[str, Any] = {
            "split": relative_tar.parts[0],
            "tar": relative_tar.as_posix(),
            "bytes": tar_path.stat().st_size,
        }
        if with_indexes:
            index_path = tar_path.with_suffix(".idx")
            first_line = index_path.read_text(encoding="utf-8").splitlines()[0].split()
            if len(first_line) != 2 or first_line[0] != "v1.2":
                raise RuntimeError(f"unexpected DALI index header in {index_path}")
            record["samples"] = int(first_line[1])
            record["index"] = index_path.relative_to(output).as_posix()
        else:
            with tarfile.open(tar_path, mode="r:") as archive:
                record["samples"] = sum(
                    member.isfile() and member.name.endswith(".labels.npy")
                    for member in archive
                )
        records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples_per_shard <= 0:
        raise ValueError("--max-samples-per-shard must be positive")
    if args.max_shard_size_gb <= 0:
        raise ValueError("--max-shard-size-gb must be positive")

    parquet = args.parquet.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    output = args.output.resolve()
    if image_root == output or image_root.is_relative_to(output):
        raise ValueError("--output must not be --image-root or one of its parents")
    if parquet == output or parquet.is_relative_to(output):
        raise ValueError("--output must not contain the input Parquet file")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; use --overwrite to replace it")

    df = pd.read_parquet(parquet)
    label_columns = validate_dataframe(df)
    image_paths, skipped = preflight_images(df, image_root, args.missing_image)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        counts = build_archives(
            df,
            image_paths,
            temporary,
            label_columns,
            args.max_samples_per_shard,
            int(args.max_shard_size_gb * 1024**3),
        )
        if not args.no_dali_index:
            create_dali_indexes(temporary)

        if skipped:
            with (temporary / "skipped.jsonl").open("w", encoding="utf-8") as stream:
                for item in skipped:
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

        shards = shard_records(temporary, with_indexes=not args.no_dali_index)
        indexed_counts: Counter[str] = Counter()
        for shard in shards:
            indexed_counts[shard["split"]] += shard["samples"]
        if any(indexed_counts[split] != counts[split] for split in SPLITS):
            raise RuntimeError(
                f"shard sample counts {dict(indexed_counts)} do not match "
                f"written counts {dict(counts)}"
            )
        manifest: dict[str, Any] = {
            "format": "paired-fundus-webdataset-v1",
            "source": {
                "parquet": str(parquet),
                "image_root": str(image_root),
                "rows": len(df),
            },
            "components": {
                "left.jpg": "JPEG-encoded left fundus image from image_path_1",
                "right.jpg": "JPEG-encoded right fundus image from image_path_2",
                "labels.npy": {
                    "dtype": "int8",
                    "shape": [len(label_columns)],
                    "columns": label_columns,
                    "missing_value": LABEL_MISSING_VALUE,
                },
                "meta.json": {
                    "encoding": "UTF-8",
                    "columns": [column for column in df.columns if column not in label_columns]
                    + ["source_row"],
                },
            },
            "splits": {split: int(counts[split]) for split in SPLITS},
            "skipped_rows": len(skipped),
            "shards": shards,
            "dali": {
                "ext": ["left.jpg", "right.jpg", "labels.npy", "meta.json"],
                "indexes_created": not args.no_dali_index,
            },
        }
        write_json(temporary / "dataset.json", manifest)

        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        manifest = build_dataset(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["splits"], ensure_ascii=False))
    print(f"skipped rows: {manifest['skipped_rows']}")
    print(f"shards: {len(manifest['shards'])}")
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
