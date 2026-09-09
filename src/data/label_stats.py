"""Read label counts directly from tar members without decoding images."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np


def count_target_values(
    webdataset_dir: str | Path, split: str, target: str
) -> dict[int, int]:
    root = Path(webdataset_dir).resolve(strict=True)
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    columns = manifest["components"]["labels.npy"]["columns"]
    if target not in columns:
        raise ValueError(f"unknown target {target!r}; available targets: {columns}")
    index = columns.index(target)
    counts = {-1: 0, 0: 0, 1: 0}
    shards = [item for item in manifest["shards"] if item["split"] == split]
    if not shards:
        raise ValueError(f"manifest has no shards for split {split!r}")

    for shard in shards:
        with tarfile.open(root / shard["tar"], "r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".labels.npy"):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read {member.name} in {shard['tar']}")
                labels = np.load(io.BytesIO(stream.read()), allow_pickle=False)
                value = int(labels[index])
                if value not in counts:
                    raise ValueError(f"invalid target value {value} in {member.name}")
                counts[value] += 1
    return counts

