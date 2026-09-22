# Split heatmap generation

Generate native-resolution heatmaps for every patient in one WebDataset split:

```bash
python scripts/heatmap/generate_split_heatmaps.py \
  --checkpoint /path/to/result_alt/best.pt \
  --data-dir /path/to/webdataset \
  --split external_validation \
  --output-file /path/to/heatmaps/result_alt.external.pt \
  --seed 2026 \
  --data-backend dali \
  --amp fp16 \
  --device cuda:1
```

Both eyes are processed together and stored in fixed order. Heatmap tensors use
layout `[patients, 2, height, width]`, where eye index 0 is left and 1 is right.
They remain at native model resolution: normally `7x7` for ResNet-50 and `16x16`
for RETFound-DINOv2. They are not converted into images or upsampled.

The default `float16` storage keeps normalized values while using about half the
space of `float32`. For 50,000 patients, the three RETFound maps require roughly
146.5 MiB before the small metadata tensors; one ResNet map requires roughly
9.3 MiB. Use `--storage-dtype float32` if full single-precision storage is needed.

The one output `.pt` contains:

- `source_row: [N]`, used to recover the parquet patient;
- `heatmaps[name]: [N, 2, H, W]`;
- `logits` and `predictions: [N, 2]`;
- compact labels, ages and sexes;
- checkpoint, split, seed, transform, threshold, eye order and method metadata.

Load it without executing arbitrary Python objects:

```python
import torch

result = torch.load("result_alt.external.pt", map_location="cpu", weights_only=True)
left = result["heatmaps"]["dinov2_gradient_rollout"][:, 0]
right = result["heatmaps"]["dinov2_gradient_rollout"][:, 1]
source_rows = result["source_row"]
```

`--transform eval` is deterministic; the seed is still recorded but will not
change pixels. `--transform train` applies the project's seeded stochastic
augmentation, so different seeds can produce different heatmaps.

CUDA uses `--amp fp16` by default. Model parameters remain FP32, the forward
pass uses autocast, gradients are scaled to avoid FP16 underflow, and Grad-CAM /
attention rollout reductions run in FP32. Use `--amp off` for a full-FP32
reference run. AMP is automatically disabled on CPU. The target V100 supports
FP16 Tensor Cores but not the BF16 path, so BF16 is intentionally not exposed.

## Queue multiple jobs

Copy `heatmap_jobs.example.txt` and put one task's arguments on each line. Blank
lines, full-line comments and trailing `#` comments are accepted. Do not specify
`--device`; the scheduler owns it.

```bash
python scripts/heatmap/schedule_heatmaps.py my_heatmap_jobs.txt \
  --max-processes-per-gpu 1
```

Each visible GPU receives up to the configured number of processes. Logs are
written under `scheduler_logs/heatmap/<time>/` by default. For an 11 GB GPU,
keep `--patient-batch-size 1`, which sends the two eyes through the model as one
batch. Larger values can improve throughput if memory permits.
