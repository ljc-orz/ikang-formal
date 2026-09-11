"""Backend-independent fixed-length balancing for binary training labels."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator

import torch


TrainingBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class BalancedBinaryLoader:
    """Resample patient classes and emit fixed-length 1:1 binary batches.

    The wrapped torchvision or DALI loader must yield adjacent left/right eyes.
    The majority class is sampled without replacement and the minority class
    with replacement. Each class contributes half the natural epoch size, so
    an even-sized source keeps exactly the same number of samples. Sampling is
    redrawn for every epoch.
    """

    def __init__(
        self,
        loader: Iterable[TrainingBatch],
        *,
        batch_size: int,
        class_counts: dict[int, int],
        seed: int,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("balanced training requires an even batch size")
        if class_counts.get(0, 0) <= 0 or class_counts.get(1, 0) <= 0:
            raise ValueError(
                "balanced training requires both label 0 and label 1 in the training set"
            )
        self.loader = loader
        self.batch_size = batch_size
        self.class_counts = {0: int(class_counts[0]), 1: int(class_counts[1])}
        self.patients_per_class = math.ceil(sum(self.class_counts.values()) / 2)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        # Two eyes for every selected patient and two equally sized classes.
        return math.ceil(4 * self.patients_per_class / self.batch_size)

    @staticmethod
    def _sample_at(batch: TrainingBatch, index: int) -> tuple[torch.Tensor, ...]:
        # DALI owns and reuses output buffers. A selected value may remain in a
        # queue while the source advances, so it needs independent storage.
        return tuple(value[index].clone() for value in batch)

    @staticmethod
    def _collate(
        samples: list[tuple[torch.Tensor, ...]], rng: random.Random
    ) -> TrainingBatch:
        rng.shuffle(samples)
        values = tuple(
            torch.stack([sample[field] for sample in samples])
            for field in range(len(samples[0]))
        )
        return values  # type: ignore[return-value]

    def __iter__(self) -> Iterator[TrainingBatch]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        extra_draws = {
            label: Counter(
                rng.randrange(count)
                for _ in range(max(0, self.patients_per_class - count))
            )
            for label, count in self.class_counts.items()
        }
        observed = {0: 0, 1: 0}
        selected = {0: 0, 1: 0}
        buffers: dict[int, list[tuple[torch.Tensor, ...]]] = {0: [], 1: []}
        samples_per_class = self.batch_size // 2

        for batch in self.loader:
            if len(batch) != 4:
                raise ValueError("balanced training expects four-tensor batches")
            labels = batch[-1]
            if labels.ndim != 1 or labels.numel() % 2:
                raise ValueError(
                    "balanced training expects adjacent left/right label pairs"
                )
            label_values = labels.detach().cpu().tolist()
            for index in range(0, len(label_values), 2):
                label = int(label_values[index])
                if label not in (0, 1) or int(label_values[index + 1]) != label:
                    raise ValueError(
                        "balanced training expects each adjacent eye pair to share "
                        "a binary label"
                    )
                position = observed[label]
                if position >= self.class_counts[label]:
                    raise RuntimeError(
                        "training loader produced more samples than its label counts"
                    )
                observed[label] += 1
                count = self.class_counts[label]
                if count <= self.patients_per_class:
                    multiplicity = 1 + extra_draws[label].get(position, 0)
                else:
                    available = count - position
                    wanted = self.patients_per_class - selected[label]
                    multiplicity = int(wanted > 0 and rng.randrange(available) < wanted)
                selected[label] += multiplicity
                if multiplicity == 0:
                    continue
                # Keep one independent copy because DALI reuses its output
                # buffers. Repeated draws can share this cached tensor until
                # torch.stack materializes the output batch.
                pair = (
                    self._sample_at(batch, index),
                    self._sample_at(batch, index + 1),
                )
                for _ in range(multiplicity):
                    buffers[label].extend(pair)

                while all(
                    len(buffers[value]) >= samples_per_class for value in (0, 1)
                ):
                    samples = (
                        buffers[0][:samples_per_class]
                        + buffers[1][:samples_per_class]
                    )
                    del buffers[0][:samples_per_class]
                    del buffers[1][:samples_per_class]
                    yield self._collate(samples, rng)

        if observed != self.class_counts:
            raise RuntimeError(
                "training loader ended before the configured label counts were observed"
            )
        if any(value != self.patients_per_class for value in selected.values()):
            raise RuntimeError("balanced sampler did not reach its class quotas")
        if len(buffers[0]) != len(buffers[1]):
            raise RuntimeError("balanced sampler finished with unequal class buffers")
        if buffers[0]:
            yield self._collate(buffers[0] + buffers[1], rng)
