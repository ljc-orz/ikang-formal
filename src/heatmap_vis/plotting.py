"""Save consistent overlays for project heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


TITLES = {
    "resnet_gradcam": "ResNet-50 Grad-CAM",
    "dinov2_gradcam": "DINOv2 Transformer Grad-CAM",
    "dinov2_gradient_rollout": "DINOv2 gradient attention rollout",
    "dinov2_cls_attention": "DINOv2 last-layer CLS attention",
}


def save_heatmap_figure(
    output: str | Path,
    image: Tensor,
    heatmaps: dict[str, Tensor],
    *,
    title: str,
    alpha: float = 0.45,
) -> None:
    """Save original + method overlays; image is RGB [3,H,W] in [0,1]."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("display image must be shaped [3, H, W]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    unknown = set(heatmaps).difference(TITLES)
    if unknown:
        raise ValueError(f"unknown heatmap names: {sorted(unknown)}")
    pixels = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    columns = 1 + len(heatmaps)
    figure, axes = plt.subplots(
        1, columns, figsize=(4.0 * columns, 4.1), squeeze=False, constrained_layout=True
    )
    axes = axes[0]
    axes[0].imshow(pixels)
    axes[0].set_title("Original")
    for axis, (name, heatmap) in zip(axes[1:], heatmaps.items()):
        values = heatmap.detach().float().cpu().numpy()
        if values.shape != pixels.shape[:2]:
            raise ValueError(
                f"heatmap {name!r} has shape {values.shape}, "
                f"expected {pixels.shape[:2]}"
            )
        axis.imshow(pixels)
        axis.imshow(values, cmap="turbo", alpha=alpha, vmin=0.0, vmax=1.0)
        axis.set_title(TITLES[name])
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(Path(output), dpi=180, bbox_inches="tight")
    plt.close(figure)
