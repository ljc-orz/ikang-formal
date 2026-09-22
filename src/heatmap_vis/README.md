# Trained-model heatmaps

This package visualizes the project's actual `FundusClassifier`, including its
fine-tuned backbone, metadata encoder and binary classification head. It supports
the two currently relevant heatmap backbones:

- `resnet50`: Grad-CAM at `model.backbone.layer4[-1]`;
- `retfound_dinov2`: Transformer Grad-CAM at the last block's `norm1`,
  gradient attention rollout over all blocks, and last-layer raw CLS attention.

The gradient objective is the complete diagnostic logit after fusing the image
with age and sex. This differs from the standalone prototype, where DINOv2 had
no trained classifier and therefore used the CLS feature norm. Raw CLS attention
still has no class gradient and should not be described as class-specific.

Run from the repository root:

```bash
python -m src.heatmap_vis \
  --checkpoint /path/to/result_alt/best.pt \
  --image /path/to/fundus.jpg \
  --age 52 \
  --sex MAN \
  --output-dir /path/to/heatmaps \
  --amp fp16 \
  --device cuda:1
```

By default, `--explanation-target predicted` explains the class selected using
the decision threshold stored in the checkpoint: the abnormal logit for a
positive prediction and its negative for a healthy prediction. Use
`--explanation-target abnormal` to always visualize evidence increasing the
abnormal logit, which is useful when comparing people regardless of prediction.

The command applies the exact deterministic evaluation preprocessing stored in
the checkpoint (direct square resize, mean and standard deviation). It writes:

- `<image>.heatmaps.png`: original image and all applicable overlays;
- `<image>.heatmaps.pt`: normalized raw heatmap tensors and prediction metadata;
- `<image>.heatmaps.json`: paths, model/target, metadata, prediction and plotting
  parameters.

All maps are independently normalized to `[0, 1]`; colors between different
methods are therefore not a shared absolute importance scale. Batch size is
fixed to one. For RETFound, fused attention is temporarily disabled so the
post-softmax attention matrices can be captured, then restored after the call.

CUDA defaults to FP16 autocast via `--amp fp16`, while Grad-CAM and attention
rollout reductions remain FP32. Use `--amp off` when an exact FP32 reference is
required. AMP is automatically disabled for CPU execution.
