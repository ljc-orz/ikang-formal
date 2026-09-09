"""PyTorch iterable loader for the paired-fundus WebDataset.

Example::

    dataset = FundusWebDataset("example/webdataset", "train", "result_alt")
    for image, age, sex, target in dataset:
        # image: uint8 RGB tensor [C, H, W]; sex: MAN=0, WOMAN=1
        ...
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset


SEX_TO_INDEX = {"MAN": 0, "WOMAN": 1}
EXPECTED_FORMAT = "paired-fundus-webdataset-v1"


class FundusWebDataset(IterableDataset):
    """Stream one eye at a time from a paired-patient WebDataset.

    Args:
        webdataset_dir: Export directory containing ``dataset.json``.
        split: Dataset split, for example ``train`` or ``external_validation``.
        result: One ``result_*`` column to return as the target.
        transform: Optional callable applied to each image tensor.
        shuffle: Shuffle patients while keeping each patient's left/right eyes
            adjacent and in that order. Defaults to ``False``.
        shuffle_buffer: Number of patient samples buffered for shuffling.
        seed: Shuffle seed.

    Each iteration returns ``(image, age, sex, result)``. ``image`` is an RGB
    ``torch.uint8`` tensor in CHW layout unless ``transform`` changes it. The
    other three values are Python integers; sex is MAN=0 and WOMAN=1. A missing
    result keeps the exporter sentinel value -1.
    """

    def __init__(
        self,
        webdataset_dir: str | Path,
        split: str,
        result: str,
        *,
        transform: Callable[[torch.Tensor], Any] | None = None,
        shuffle: bool = False,
        shuffle_buffer: int = 1_000,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.webdataset_dir = Path(webdataset_dir).resolve(strict=True)
        self.split = split
        self.result = result
        self.transform = transform
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")

        manifest_path = self.webdataset_dir / "dataset.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != EXPECTED_FORMAT:
            raise ValueError(f"unsupported dataset format: {manifest.get('format')!r}")

        splits = manifest.get("splits", {})
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; available splits: {list(splits)}")

        label_spec = manifest.get("components", {}).get("labels.npy", {})
        self.result_names = tuple(label_spec.get("columns", ()))
        if result not in self.result_names:
            raise ValueError(
                f"unknown result {result!r}; available results: {list(self.result_names)}"
            )
        self.result_index = self.result_names.index(result)
        self.patient_count = int(splits[split])

        shard_records = [
            shard for shard in manifest.get("shards", []) if shard.get("split") == split
        ]
        if not shard_records:
            raise ValueError(f"manifest has no shards for split {split!r}")
        self.tar_paths = tuple(
            str(self.webdataset_dir / str(shard["tar"])) for shard in shard_records
        )
        missing_tars = [path for path in self.tar_paths if not Path(path).is_file()]
        if missing_tars:
            raise FileNotFoundError(f"tar shard(s) do not exist: {missing_tars}")

    def __len__(self) -> int:
        """Number of eye samples, twice the number of complete patients."""
        return self.patient_count * 2

    def _patients(self) -> wds.DataPipeline:
        dataset = wds.WebDataset(
            list(self.tar_paths),
            shardshuffle=self.shuffle_buffer if self.shuffle else False,
            detshuffle=self.shuffle,
            seed=self.seed,
            nodesplitter=wds.split_by_node,
            empty_check=False,
        )
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer, seed=self.seed)
        return dataset.decode("torchrgb8").to_tuple(
            "left.jpg", "right.jpg", "labels.npy", "meta.json"
        )

    def __iter__(self) -> Iterator[tuple[Any, int, int, int]]:
        for left, right, labels, metadata in self._patients():
            age = self._parse_age(metadata)
            sex = self._parse_sex(metadata)
            target = self._parse_result(labels)
            for image in (left, right):
                if self.transform is not None:
                    image = self.transform(image)
                yield image, age, sex, target

    @staticmethod
    def _parse_age(metadata: dict[str, Any]) -> int:
        age = metadata.get("examage")
        if age is None or isinstance(age, bool):
            raise ValueError(f"invalid examage in meta.json: {age!r}")
        try:
            return int(age)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid examage in meta.json: {age!r}") from exc

    @staticmethod
    def _parse_sex(metadata: dict[str, Any]) -> int:
        sex = metadata.get("usex")
        try:
            return SEX_TO_INDEX[sex]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid usex in meta.json: {sex!r}; expected MAN or WOMAN"
            ) from exc

    def _parse_result(self, labels: np.ndarray) -> int:
        if labels.ndim != 1 or labels.shape[0] != len(self.result_names):
            raise ValueError(
                f"invalid labels.npy shape {labels.shape}; "
                f"expected ({len(self.result_names)},)"
            )
        return int(labels[self.result_index])
