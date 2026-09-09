"""Image transforms used by the V1 training and evaluation pipelines."""

from __future__ import annotations

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def build_train_transform(
    image_size: int,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.90, 1.00),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.03, 0.03),
                        scale=(0.95, 1.05),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.7,
            ),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.10,
                        contrast=0.10,
                        saturation=0.05,
                        hue=0.0,
                    )
                ],
                p=0.5,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.1),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(
    image_size: int,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> T.Compose:
    return T.Compose(
        [
            T.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=mean, std=std),
        ]
    )
