"""Patient-level paired-eye WebDataset used for evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .fundus_webdataset import FundusWebDataset


class PairedFundusWebDataset(FundusWebDataset):
    """Return both eyes from each patient without separating the pair.

    Each item is ``(left, right, age, sex, target, patient_id)``. The target can
    be missing only when ``skip_missing_target=False``.
    """

    def __init__(
        self,
        webdataset_dir: str | Path,
        split: str,
        result: str,
        **kwargs: Any,
    ) -> None:
        # Evaluation order must be stable, so paired data is never shuffled.
        kwargs["shuffle"] = False
        super().__init__(webdataset_dir, split, result, **kwargs)

    def __len__(self) -> int:
        return self.patient_count

    def __iter__(self) -> Iterator[tuple[Any, Any, int, int, int, str]]:
        for left, right, labels, metadata in self._patients():
            age = self._parse_age(metadata)
            sex = self._parse_sex(metadata)
            target = self._parse_result(labels)
            if self.skip_missing_target and target < 0:
                continue
            if self.transform is not None:
                left = self.transform(left)
                right = self.transform(right)
            patient_id = metadata.get("uid")
            if patient_id is None:
                patient_id = metadata.get("id", metadata.get("source_row", ""))
            yield left, right, age, sex, target, str(patient_id)

