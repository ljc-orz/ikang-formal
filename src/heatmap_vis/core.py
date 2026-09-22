"""Gradient heatmaps for the actual metadata-fused project models."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model import FundusClassifier


EXPLANATION_TARGETS = ("predicted", "abnormal")
AMP_GRADIENT_SCALE = 1024.0


@dataclass(frozen=True)
class HeatmapResult:
    heatmaps: dict[str, Tensor]
    logit: float
    probability: float
    predicted_label: int
    decision_threshold: float
    explanation_target: str
    objective: float


@dataclass(frozen=True)
class BatchHeatmapResult:
    heatmaps: dict[str, Tensor]
    logits: Tensor
    probabilities: Tensor
    predicted_labels: Tensor
    decision_threshold: float
    explanation_target: str
    objectives: Tensor


def normalize_heatmap(heatmap: Tensor, eps: float = 1e-8) -> Tensor:
    heatmap = heatmap.detach().float()
    heatmap = heatmap - heatmap.amin()
    maximum = heatmap.amax()
    if maximum <= eps:
        return torch.zeros_like(heatmap)
    return (heatmap / maximum).clamp_(0.0, 1.0)


def _finish_heatmaps(
    heatmaps: Tensor, output_size: tuple[int, int] | None
) -> Tensor:
    """Normalize every sample independently and optionally resize it."""
    values = heatmaps.detach().float()
    if values.ndim != 3:
        raise ValueError(f"expected heatmaps [B,H,W], got {tuple(values.shape)}")
    if output_size is not None:
        values = F.interpolate(
            values[:, None],
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    flat = values.flatten(1)
    minimum = flat.amin(dim=1).reshape(-1, 1, 1)
    values = values - minimum
    maximum = values.flatten(1).amax(dim=1).reshape(-1, 1, 1)
    values = torch.where(
        maximum > 1e-8,
        values / maximum.clamp_min(1e-8),
        torch.zeros_like(values),
    )
    return values.clamp_(0.0, 1.0).cpu()


@contextmanager
def _capture_activation(module: nn.Module):
    captured: dict[str, Tensor] = {}

    def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
        captured["activation"] = output

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def _capture_attentions(blocks: Iterable[nn.Module]):
    captures: list[dict[str, Tensor]] = []
    handles = []
    fused_values = []
    for block in blocks:
        attention_module = block.attn
        fused_values.append(bool(attention_module.fused_attn))
        attention_module.fused_attn = False
        captured: dict[str, Tensor] = {}
        captures.append(captured)

        def hook(
            _module: nn.Module,
            _inputs: Any,
            output: Tensor,
            destination: dict[str, Tensor] = captured,
        ) -> None:
            destination["attention"] = output

        handles.append(attention_module.attn_drop.register_forward_hook(hook))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        for block, fused in zip(blocks, fused_values):
            block.attn.fused_attn = fused


def reshape_patch_tokens(
    tokens: Tensor, grid_size: Sequence[int], prefix_tokens: int
) -> Tensor:
    """Change [B, prefix + H*W, C] into [B, C, H, W]."""
    height, width = int(grid_size[0]), int(grid_size[1])
    patches = tokens[:, prefix_tokens:, :]
    expected = height * width
    if patches.shape[1] != expected:
        raise ValueError(
            f"token/grid mismatch: got {patches.shape[1]} patch tokens, "
            f"expected {height}*{width}={expected}"
        )
    return patches.reshape(tokens.shape[0], height, width, tokens.shape[-1]).permute(
        0, 3, 1, 2
    )


def gradient_attention_rollout(
    attentions: Iterable[Tensor], gradients: Iterable[Tensor]
) -> Tensor:
    """Return one full gradient-weighted rollout matrix per sample."""
    rollout = None
    for attention, gradient in zip(attentions, gradients):
        matrix = (attention.float() * gradient.float()).relu().mean(dim=1)
        matrix = matrix + torch.eye(
            matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
        ).unsqueeze(0)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = matrix if rollout is None else matrix @ rollout
    if rollout is None:
        raise ValueError("at least one attention matrix is required")
    return rollout


def _validate_inputs(
    image: Tensor,
    age: Tensor,
    sex: Tensor,
    decision_threshold: float,
    explanation_target: str,
) -> None:
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3:
        raise ValueError("images must be shaped [B, 3, H, W] with B >= 1")
    if age.shape != (image.shape[0],) or sex.shape != (image.shape[0],):
        raise ValueError("age and sex must each contain one value per image")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be in [0, 1]")
    if explanation_target not in EXPLANATION_TARGETS:
        raise ValueError(
            f"unknown explanation_target {explanation_target!r}; "
            f"expected one of {EXPLANATION_TARGETS}"
        )


def _objectives(
    logit: Tensor, decision_threshold: float, explanation_target: str
) -> tuple[Tensor, Tensor, Tensor]:
    # Keep the scalar objective and threshold comparison in FP32 even when the
    # backbone forward uses autocast.
    logit_float = logit.float()
    probability = logit_float.sigmoid()
    predicted_labels = probability.detach().ge(decision_threshold).to(torch.uint8)
    if explanation_target == "abnormal":
        objectives = logit_float
    else:
        # A binary classifier has one abnormal logit. Its negative is the
        # corresponding class-0 direction when explaining a healthy prediction.
        objectives = torch.where(predicted_labels.bool(), logit_float, -logit_float)
    return objectives, probability, predicted_labels


def _autocast(image: Tensor, amp_dtype: torch.dtype | None):
    if image.device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _gradient_scale(image: Tensor, amp_dtype: torch.dtype | None) -> float:
    return AMP_GRADIENT_SCALE if image.device.type == "cuda" and amp_dtype else 1.0


def _resnet_heatmaps(
    model: FundusClassifier,
    image: Tensor,
    age: Tensor,
    sex: Tensor,
    decision_threshold: float,
    explanation_target: str,
    output_size: tuple[int, int] | None,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    target_layer = model.backbone.layer4[-1]
    with _capture_activation(target_layer) as captured:
        with _autocast(image, amp_dtype):
            logit = model(image, age, sex)
        objectives, probability, predicted_labels = _objectives(
            logit, decision_threshold, explanation_target
        )
        activation = captured.get("activation")
        if activation is None:
            raise RuntimeError("failed to capture ResNet activation")
        scale = _gradient_scale(image, amp_dtype)
        gradient = torch.autograd.grad(objectives.sum() * scale, activation)[0]
        gradient = gradient.float().div_(scale)
    weights = gradient.float().mean(dim=(-2, -1), keepdim=True)
    cam = (weights * activation.float()).sum(dim=1).relu()
    return (
        {"resnet_gradcam": _finish_heatmaps(cam, output_size)},
        logit,
        probability,
        predicted_labels,
        objectives,
    )


def _dinov2_heatmaps(
    model: FundusClassifier,
    image: Tensor,
    age: Tensor,
    sex: Tensor,
    decision_threshold: float,
    explanation_target: str,
    output_size: tuple[int, int] | None,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    backbone = model.backbone
    blocks = backbone.blocks
    target_layer = blocks[-1].norm1
    prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
    grid_size = tuple(backbone.patch_embed.grid_size)

    with _capture_activation(target_layer) as activation_capture, _capture_attentions(
        blocks
    ) as attention_captures:
        with _autocast(image, amp_dtype):
            logit = model(image, age, sex)
        objectives, probability, predicted_labels = _objectives(
            logit, decision_threshold, explanation_target
        )
        activation = activation_capture.get("activation")
        attentions = [captured.get("attention") for captured in attention_captures]
        if activation is None:
            raise RuntimeError("failed to capture DINOv2 activation")
        if any(value is None for value in attentions):
            raise RuntimeError("failed to capture all DINOv2 attention matrices")
        scale = _gradient_scale(image, amp_dtype)
        autograd_gradients = torch.autograd.grad(
            objectives.sum() * scale,
            (activation, *attentions),
        )
        gradient = autograd_gradients[0].float().div_(scale)
        attention_gradients = tuple(
            value.float().div_(scale) for value in autograd_gradients[1:]
        )

    activations = reshape_patch_tokens(activation, grid_size, prefix_tokens)
    patch_gradients = reshape_patch_tokens(gradient, grid_size, prefix_tokens)
    weights = patch_gradients.mean(dim=(-2, -1), keepdim=True)
    gradcam = (weights * activations.float()).sum(dim=1).relu()

    rollout = gradient_attention_rollout(attentions, attention_gradients)
    rollout_map = rollout[:, 0, prefix_tokens:].reshape(-1, *grid_size)
    final_attention = attentions[-1].float().mean(dim=1)
    cls_attention = final_attention[:, 0, prefix_tokens:].reshape(-1, *grid_size)
    return (
        {
            "dinov2_gradcam": _finish_heatmaps(gradcam, output_size),
            "dinov2_gradient_rollout": _finish_heatmaps(rollout_map, output_size),
            "dinov2_cls_attention": _finish_heatmaps(cls_attention, output_size),
        },
        logit,
        probability,
        predicted_labels,
        objectives,
    )


def generate_heatmaps_batch(
    model: FundusClassifier,
    image: Tensor,
    age: Tensor,
    sex: Tensor,
    *,
    decision_threshold: float = 0.5,
    explanation_target: str = "predicted",
    native_resolution: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> BatchHeatmapResult:
    """Explain a batch using the complete fine-tuned diagnostic model.

    ``predicted`` explains the checkpoint-threshold class: positive logit for an
    abnormal prediction and negative logit for a healthy prediction.
    ``abnormal`` always explains evidence that increases the abnormal logit.
    """
    _validate_inputs(image, age, sex, decision_threshold, explanation_target)
    if amp_dtype not in (None, torch.float16):
        raise ValueError("heatmap AMP supports only torch.float16")
    model.eval()
    model.zero_grad(set_to_none=True)
    # The RETFound base weights are frozen. Requiring an image gradient keeps
    # the graph through every transformer block so rollout can use all layers.
    image = image.detach().requires_grad_(True)
    output_size = None if native_resolution else tuple(image.shape[-2:])
    if model.backbone_type == "resnet50":
        values = _resnet_heatmaps(
            model,
            image,
            age,
            sex,
            decision_threshold,
            explanation_target,
            output_size,
            amp_dtype,
        )
    elif model.backbone_type == "retfound_dinov2":
        values = _dinov2_heatmaps(
            model,
            image,
            age,
            sex,
            decision_threshold,
            explanation_target,
            output_size,
            amp_dtype,
        )
    else:
        raise ValueError(
            "heatmap generation supports 'resnet50' and 'retfound_dinov2', "
            f"not {model.backbone_type!r}"
        )
    heatmaps, logits, probabilities, predicted_labels, objectives = values
    return BatchHeatmapResult(
        heatmaps=heatmaps,
        logits=logits.detach().float().cpu(),
        probabilities=probabilities.detach().float().cpu(),
        predicted_labels=predicted_labels.detach().cpu(),
        decision_threshold=float(decision_threshold),
        explanation_target=explanation_target,
        objectives=objectives.detach().float().cpu(),
    )


def generate_heatmaps(
    model: FundusClassifier,
    image: Tensor,
    age: Tensor,
    sex: Tensor,
    *,
    decision_threshold: float = 0.5,
    explanation_target: str = "predicted",
    native_resolution: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> HeatmapResult:
    """Single-image convenience wrapper around :func:`generate_heatmaps_batch`."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("generate_heatmaps requires exactly one image")
    batch = generate_heatmaps_batch(
        model,
        image,
        age,
        sex,
        decision_threshold=decision_threshold,
        explanation_target=explanation_target,
        native_resolution=native_resolution,
        amp_dtype=amp_dtype,
    )
    return HeatmapResult(
        heatmaps={name: values[0] for name, values in batch.heatmaps.items()},
        logit=float(batch.logits[0]),
        probability=float(batch.probabilities[0]),
        predicted_label=int(batch.predicted_labels[0]),
        decision_threshold=batch.decision_threshold,
        explanation_target=batch.explanation_target,
        objective=float(batch.objectives[0]),
    )
