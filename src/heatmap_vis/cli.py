"""Command-line heatmap generation for one eye and one trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data import build_eval_transform

from .checkpoint import load_fundus_checkpoint
from .core import EXPLANATION_TARGETS, generate_heatmaps
from .plotting import save_heatmap_figure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument(
        "--sex",
        choices=("MAN", "WOMAN", "0", "1"),
        required=True,
        help="Training encoding is MAN=0 and WOMAN=1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--explanation-target",
        choices=EXPLANATION_TARGETS,
        default="predicted",
        help="Explain the thresholded predicted class, or always abnormal evidence",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--amp",
        choices=("off", "fp16"),
        default="fp16",
        help="CUDA forward precision (default: fp16; automatically off on CPU)",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sex_index(value: str) -> int:
    return 0 if value in ("MAN", "0") else 1


def run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    amp_dtype = (
        torch.float16 if device.type == "cuda" and args.amp == "fp16" else None
    )
    loaded = load_fundus_checkpoint(args.checkpoint, device)
    image_path = args.image.expanduser().resolve(strict=True)
    raw_image = decode_image(str(image_path), mode=ImageReadMode.RGB)
    transform = build_eval_transform(loaded.image_size, loaded.mean, loaded.std)
    model_input = transform(raw_image).unsqueeze(0).to(device)
    display_image = TF.resize(
        raw_image,
        [loaded.image_size, loaded.image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).float().div(255.0)
    age = torch.tensor([args.age], dtype=torch.float32, device=device)
    sex = torch.tensor([_sex_index(args.sex)], dtype=torch.float32, device=device)
    result = generate_heatmaps(
        loaded.model,
        model_input,
        age,
        sex,
        decision_threshold=loaded.threshold,
        explanation_target=args.explanation_target,
        amp_dtype=amp_dtype,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    figure_path = args.output_dir / f"{stem}.heatmaps.png"
    tensor_path = args.output_dir / f"{stem}.heatmaps.pt"
    manifest_path = args.output_dir / f"{stem}.heatmaps.json"
    title = (
        f"{loaded.target}: p(abnormal)={result.probability:.4f}, "
        f"prediction={result.predicted_label}, age={args.age:g}, sex={args.sex}"
    )
    save_heatmap_figure(
        figure_path,
        display_image,
        result.heatmaps,
        title=title,
        alpha=args.alpha,
    )
    torch.save(
        {
            "heatmaps": result.heatmaps,
            "logit": result.logit,
            "probability": result.probability,
            "predicted_label": result.predicted_label,
            "decision_threshold": result.decision_threshold,
            "explanation_target": result.explanation_target,
            "target": loaded.target,
            "backbone": loaded.backbone_type,
            "image": str(image_path),
            "age": float(args.age),
            "sex": _sex_index(args.sex),
            "amp": "fp16" if amp_dtype is not None else "off",
        },
        tensor_path,
    )
    manifest = {
        "checkpoint": str(loaded.checkpoint_path),
        "image": str(image_path),
        "target": loaded.target,
        "backbone": loaded.backbone_type,
        "age": float(args.age),
        "sex": _sex_index(args.sex),
        "image_size": loaded.image_size,
        "mean": loaded.mean,
        "std": loaded.std,
        "logit": result.logit,
        "probability": result.probability,
        "decision_threshold": result.decision_threshold,
        "predicted_label": result.predicted_label,
        "explanation_target": result.explanation_target,
        "gradient_objective": result.objective,
        "methods": list(result.heatmaps),
        "normalization": "independent min-max per heatmap",
        "colormap": "turbo",
        "overlay_alpha": float(args.alpha),
        "amp": "fp16" if amp_dtype is not None else "off",
        "figure": str(figure_path.resolve()),
        "tensor_file": str(tensor_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved target={loaded.target} backbone={loaded.backbone_type} "
        f"probability={result.probability:.6f} prediction={result.predicted_label} "
        f"output={figure_path.resolve()}",
        flush=True,
    )
    return manifest


def main() -> int:
    run(parse_args())
    return 0
