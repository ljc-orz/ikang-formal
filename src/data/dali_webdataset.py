"""Optional NVIDIA DALI WebDataset loader with GPU JPEG decode and augmentation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch


COMPONENTS = ("left.jpg", "right.jpg", "labels.npy", "meta.json")
SEX_TO_INDEX = {"MAN": 0, "WOMAN": 1}
LoaderMode = Literal["eyes", "pairs"]


@dataclass(frozen=True)
class DaliDatasetSpec:
    root: Path
    split: str
    tar_paths: list[str]
    index_paths: list[str]
    label_names: list[str]
    sample_count: int


def _require_dali():
    try:
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
    except ImportError as exc:
        raise RuntimeError(
            "the DALI data backend was requested but NVIDIA DALI is not installed; "
            "for CUDA 12 run: pip install -r requirements-dali-cu12.txt"
        ) from exc
    return fn, pipeline_def, types, LastBatchPolicy, DALIGenericIterator


def read_dali_dataset_spec(dataset_root: str | Path, split: str) -> DaliDatasetSpec:
    root = Path(dataset_root).resolve(strict=True)
    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "paired-fundus-webdataset-v1":
        raise ValueError(f"unsupported dataset format: {manifest.get('format')!r}")
    if split not in manifest.get("splits", {}):
        raise ValueError(
            f"unknown split {split!r}; available splits: {list(manifest.get('splits', {}))}"
        )

    components = manifest.get("components", {})
    missing_components = [name for name in COMPONENTS if name not in components]
    if missing_components:
        raise ValueError(f"manifest is missing components: {missing_components}")
    shards = [item for item in manifest.get("shards", []) if item.get("split") == split]
    if not shards:
        raise ValueError(f"manifest has no shards for split {split!r}")
    missing_index_records = [item["tar"] for item in shards if "index" not in item]
    if missing_index_records:
        raise RuntimeError(
            "DALI backend requires .idx sidecars; missing index records for: "
            + ", ".join(missing_index_records)
        )

    tar_paths = [str(root / item["tar"]) for item in shards]
    index_paths = [str(root / item["index"]) for item in shards]
    missing_files = [
        path for path in (*tar_paths, *index_paths) if not Path(path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"DALI shard/index file(s) do not exist: {missing_files}")
    return DaliDatasetSpec(
        root=root,
        split=split,
        tar_paths=tar_paths,
        index_paths=index_paths,
        label_names=list(components["labels.npy"]["columns"]),
        sample_count=int(manifest["splits"][split]),
    )


def _create_pipeline(
    spec: DaliDatasetSpec,
    *,
    batch_size: int,
    num_threads: int,
    device_id: int,
    augment: bool,
    shuffle: bool,
    image_size: int,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
    seed: int,
    dont_use_mmap: bool,
    prefetch_queue_depth: int,
):
    fn, pipeline_def, types, _, _ = _require_dali()
    pixel_mean = [255.0 * float(value) for value in mean]
    pixel_std = [255.0 * float(value) for value in std]

    def decode_and_augment(encoded: Any, seed_offset: int):
        # Keep this sequence and its parameter ranges aligned with
        # build_train_transform in transforms.py. The two libraries use
        # different random-number generators, so individual images will not be
        # pixel-identical, but they receive the same transforms and sampling
        # distributions.
        image = fn.decoders.image_random_crop(
            encoded,
            device="mixed",
            output_type=types.RGB,
            random_area=[0.90, 1.00],
            random_aspect_ratio=[0.95, 1.05],
            jpeg_fancy_upsampling=True,
            seed=seed + seed_offset,
        )
        image = fn.resize(
            image,
            device="gpu",
            resize_x=image_size,
            resize_y=image_size,
            interp_type=types.INTERP_LINEAR,
            antialias=True,
        )

        mirror = fn.random.coin_flip(
            probability=0.5, seed=seed + seed_offset + 1
        )
        image = fn.flip(image, device="gpu", horizontal=mirror)

        affine_enabled = fn.cast(
            fn.random.coin_flip(probability=0.7, seed=seed + seed_offset + 2),
            dtype=types.FLOAT,
        )
        angle = affine_enabled * fn.random.uniform(
            range=[-10.0, 10.0], seed=seed + seed_offset + 3
        )
        # torchvision rounds translations to integer pixels. Asking DALI's
        # uniform generator for INT32 values applies the same rounding.
        translation = affine_enabled * fn.random.uniform(
            range=[-0.03 * image_size, 0.03 * image_size],
            shape=[2],
            dtype=types.INT32,
            seed=seed + seed_offset + 4,
        )
        scale = 1.0 + affine_enabled * fn.random.uniform(
            range=[-0.05, 0.05], seed=seed + seed_offset + 5
        )
        center = [(image_size - 1) / 2.0, (image_size - 1) / 2.0]
        affine = fn.transforms.scale(scale=fn.stack(scale, scale), center=center)
        affine = fn.transforms.rotation(affine, angle=angle, center=center)
        affine = fn.transforms.translation(affine, offset=translation)
        image = fn.warp_affine(
            image,
            affine,
            device="gpu",
            size=[image_size, image_size],
            inverse_map=False,
            interp_type=types.INTERP_LINEAR,
            fill_value=0,
        )

        color_enabled = fn.cast(
            fn.random.coin_flip(probability=0.5, seed=seed + seed_offset + 6),
            dtype=types.FLOAT,
        )
        brightness = 1.0 + color_enabled * fn.random.uniform(
            range=[-0.10, 0.10], seed=seed + seed_offset + 7
        )
        contrast = 1.0 + color_enabled * fn.random.uniform(
            range=[-0.10, 0.10], seed=seed + seed_offset + 8
        )
        saturation = 1.0 + color_enabled * fn.random.uniform(
            range=[-0.05, 0.05], seed=seed + seed_offset + 9
        )
        image = fn.color_twist(
            image,
            device="gpu",
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=0.0,
        )

        blur_enabled = fn.cast(
            fn.random.coin_flip(probability=0.1, seed=seed + seed_offset + 10),
            dtype=types.FLOAT,
        )
        # sigma=1e-4 is effectively an identity 3x3 kernel when blur is disabled.
        sigma = 0.0001 + blur_enabled * fn.random.uniform(
            range=[0.0999, 0.7999], seed=seed + seed_offset + 11
        )
        image = fn.gaussian_blur(
            image, device="gpu", window_size=3, sigma=sigma
        )
        return fn.crop_mirror_normalize(
            image,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=pixel_mean,
            std=pixel_std,
        )

    def decode_for_evaluation(encoded: Any):
        image = fn.decoders.image(
            encoded,
            device="mixed",
            output_type=types.RGB,
            jpeg_fancy_upsampling=True,
        )
        image = fn.resize(
            image,
            device="gpu",
            resize_x=image_size,
            resize_y=image_size,
            interp_type=types.INTERP_LINEAR,
            antialias=True,
        )
        return fn.crop_mirror_normalize(
            image,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=pixel_mean,
            std=pixel_std,
        )

    @pipeline_def
    def paired_pipeline():
        left_raw, right_raw, labels_raw, metadata = fn.readers.webdataset(
            paths=spec.tar_paths,
            index_paths=spec.index_paths,
            ext=list(COMPONENTS),
            missing_component_behavior="error",
            random_shuffle=shuffle,
            seed=seed,
            dont_use_mmap=dont_use_mmap,
            pad_last_batch=True,
            name="Reader",
        )
        if augment:
            left = decode_and_augment(left_raw, 100)
            right = decode_and_augment(right_raw, 200)
        else:
            left = decode_for_evaluation(left_raw)
            right = decode_for_evaluation(right_raw)
        labels = fn.decoders.numpy(labels_raw)
        # JSON strings have variable lengths. Padding makes this output dense so
        # DALIGenericIterator can transfer it to a PyTorch uint8 tensor.
        metadata = fn.pad(metadata, fill_value=0)
        return left, right, labels, metadata

    pipeline = paired_pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    pipeline.build()
    return pipeline


def _parse_metadata_tensor(value: torch.Tensor) -> list[dict[str, Any]]:
    rows = value.detach().cpu().numpy()
    result: list[dict[str, Any]] = []
    for row in rows:
        encoded = np.asarray(row, dtype=np.uint8).tobytes().rstrip(b"\0")
        result.append(json.loads(encoded.decode("utf-8")))
    return result


class DaliFundusLoader:
    """Yield model-ready PyTorch batches produced by a DALI pipeline."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        target: str,
        *,
        mode: LoaderMode,
        batch_size: int,
        num_threads: int,
        device_id: int,
        image_size: int,
        mean: list[float] | tuple[float, ...],
        std: list[float] | tuple[float, ...],
        seed: int,
        skip_missing_target: bool,
        identity: Literal["patient_id", "source_row"] = "patient_id",
        augment: bool | None = None,
        dont_use_mmap: bool = False,
        prefetch_queue_depth: int = 2,
    ) -> None:
        if mode not in ("eyes", "pairs"):
            raise ValueError(f"unknown DALI loader mode: {mode!r}")
        if identity not in ("patient_id", "source_row"):
            raise ValueError(f"unknown patient identity mode: {identity!r}")
        if batch_size <= 0 or num_threads <= 0 or prefetch_queue_depth <= 0:
            raise ValueError("DALI batch size, threads and prefetch depth must be positive")
        if mode == "eyes" and batch_size % 2:
            raise ValueError("DALI training requires an even --batch-size")
        self.spec = read_dali_dataset_spec(dataset_root, split)
        if target not in self.spec.label_names:
            raise ValueError(
                f"unknown target {target!r}; available targets: {self.spec.label_names}"
            )
        self.target_index = self.spec.label_names.index(target)
        self.mode = mode
        self.identity = identity
        self.skip_missing_target = skip_missing_target
        self.patient_batch_size = batch_size // 2 if mode == "eyes" else batch_size
        self.pipeline = _create_pipeline(
            self.spec,
            batch_size=self.patient_batch_size,
            num_threads=num_threads,
            device_id=device_id,
            augment=mode == "eyes" if augment is None else augment,
            shuffle=mode == "eyes",
            image_size=image_size,
            mean=mean,
            std=std,
            seed=seed,
            dont_use_mmap=dont_use_mmap,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        _, _, _, LastBatchPolicy, DALIGenericIterator = _require_dali()
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            output_map=["left", "right", "labels", "metadata"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __len__(self) -> int:
        return math.ceil(self.spec.sample_count / self.patient_batch_size)

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        for pipeline_outputs in self.iterator:
            batch = pipeline_outputs[0]
            left = batch["left"]
            right = batch["right"]
            labels = batch["labels"][:, self.target_index].to(dtype=torch.int64)
            metadata = _parse_metadata_tensor(batch["metadata"])
            valid_indices = [
                index
                for index, value in enumerate(labels.tolist())
                if not self.skip_missing_target or value >= 0
            ]
            if not valid_indices:
                continue
            device_indices = torch.tensor(valid_indices, device=left.device)
            left = left.index_select(0, device_indices)
            right = right.index_select(0, device_indices)
            target = labels[valid_indices].to(device=left.device)
            selected_metadata = [metadata[index] for index in valid_indices]
            ages = torch.tensor(
                [int(row["examage"]) for row in selected_metadata],
                device=left.device,
            )
            try:
                sexes = torch.tensor(
                    [SEX_TO_INDEX[row["usex"]] for row in selected_metadata],
                    device=left.device,
                )
            except KeyError as exc:
                raise ValueError(
                    f"invalid usex in meta.json: {exc.args[0]!r}; expected MAN or WOMAN"
                ) from exc

            if self.mode == "eyes":
                yield (
                    torch.stack((left, right), dim=1).flatten(0, 1),
                    ages.repeat_interleave(2),
                    sexes.repeat_interleave(2),
                    target.repeat_interleave(2),
                )
            else:
                if self.identity == "source_row":
                    identities = []
                    for row in selected_metadata:
                        source_row = row.get("source_row")
                        if isinstance(source_row, bool) or not isinstance(source_row, int):
                            raise ValueError(
                                f"invalid source_row in meta.json: {source_row!r}"
                            )
                        identities.append(source_row)
                else:
                    identities = [
                        str(row.get("uid", row.get("id", row.get("source_row", ""))))
                        for row in selected_metadata
                    ]
                yield left, right, ages, sexes, target, identities
