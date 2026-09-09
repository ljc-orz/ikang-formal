下面是一套可以直接作为首个可靠 baseline 的配置，适合 `timm ConvNeXt-Tiny + 单眼彩照 + 年龄/性别 + 二分类 logit`。

## 推荐总配置

```yaml
model:
  name: convnext_tiny.fb_in22k_ft_in1k
  pretrained: true
  image_size: 224
  drop_path_rate: 0.1
  metadata_hidden_dim: 16
  classifier_dropout: 0.2
  num_classes: 1

training:
  max_epochs: 30
  global_batch_size: 128
  optimizer: AdamW
  mixed_precision: fp16
  gradient_clip_norm: 1.0
  early_stopping_patience: 6

loss:
  name: BCEWithLogitsLoss
  pos_weight: auto

scheduler:
  name: cosine
  warmup_epochs: 1
  min_lr: 1.0e-6

checkpoint:
  primary_metric: val_auprc
  secondary_metric: val_auroc
  mode: max
```

## 分阶段微调

建议在每次解冻后重新创建 optimizer。

| 阶段  | Epoch | 训练参数                      |                         学习率 |
| --- | ----: | ------------------------- | --------------------------: |
| 阶段一 |   0–1 | metadata 分支、融合分类头         |                      `1e-3` |
| 阶段二 |   2–7 | ConvNeXt 最后两个 stage + 融合头 | backbone `1e-4`，head `3e-4` |
| 阶段三 |  8–29 | 全部参数                      |                       分层学习率 |

阶段三推荐：

```text
stem + stage 1:    1e-5
stage 2:           2e-5
stage 3:           5e-5
stage 4:           1e-4
metadata + head:   3e-4
```

阶段三使用：

* 1 epoch linear warm-up
* 之后 cosine decay
* 最低学习率 `1e-6`
* `AdamW`
* `weight_decay=0.05`
* bias、LayerNorm 参数不使用 weight decay

`timm` 的优化器支持自动构造 weight-decay 参数组以及 layer-wise learning-rate decay。([huggingface.co][1])

如果不想手动设置每个 stage，也可以尝试：

```python
optimizer = timm.optim.create_optimizer_v2(
    model,
    opt="adamw",
    lr=3e-4,
    weight_decay=0.05,
    layer_decay=0.75,
)
```

不过因为你的模型还有 metadata 分支，手动参数分组会更可控。

## Batch size

推荐全局 batch size 为 `128`：

$$
B_{\text{global}}
=
B_{\text{per GPU}}
\times GPU数量
\times 梯度累积次数
$$

参考配置：

| GPU              |   224 分辨率 |   384 分辨率 |
| ---------------- | --------: | --------: |
| RTX 2080 Ti 11GB | 24–32/GPU |   6–8/GPU |
| RTX 3090 24GB    | 48–64/GPU | 16–24/GPU |
| A100/PG199       | 64 以上/GPU | 24–48/GPU |

例如两张 3090：

```yaml
batch_size_per_gpu: 64
gradient_accumulation_steps: 1
global_batch_size: 128
```

如果使用六张 GPU，不必把全局 batch 提高到 384，可以使用每卡 20～24，保持全局 batch 在 128 左右。

## Epoch 数量调整

这里应按独立患者数量，而不是图片数判断：

|          训练患者数 | 最大 epoch |
| -------------: | -------: |
|      `< 5,000` |    40–50 |
| `5,000–50,000` |    25–35 |
|     `> 50,000` |    15–25 |

默认先跑 30 epoch，使用 early stopping。不要因为训练 loss 仍在下降就继续训练，应以验证集 AUPRC/AUROC 为准。

## 类别不平衡

默认使用：

```python
pos_weight = num_negative / num_positive
```

为了避免极端权重，建议限制：

```python
pos_weight = min(num_negative / num_positive, 10.0)
```

然后：

```python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=device)
)
```

建议：

* 阳性比例超过 20%：先使用普通 BCE。
* 阳性比例 5%～20%：使用 `pos_weight`。
* 阳性比例低于 5%：使用 `pos_weight`，并重点观察 AUPRC。
* 不要同时使用较大的 `pos_weight` 和 `WeightedRandomSampler`，否则会重复强化阳性样本。
* 第一版不使用 Focal Loss，先建立 BCE baseline。

## 混合精度与稳定性

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    logits = model(images, ages, sexes)
    loss = criterion(logits, labels)
```

推荐：

* RTX 2080 Ti / 3090：FP16 AMP。
* A100：BF16 优先。
* 梯度裁剪：`max_norm=1.0`。
* 不需要额外 label smoothing。
* 第一版不使用 EMA、MixUp、CutMix。

## 模型选择与早停

每个 epoch 至少记录：

```text
train_loss
val_loss
val_auroc
val_auprc
val_sensitivity
val_specificity
```

如果阳性类别较少：

```yaml
save_best_by: val_auprc
early_stopping:
  metric: val_auprc
  patience: 6
  min_delta: 0.001
```

最终阈值不要默认固定为 `0.5`。应在验证集上按照具体目标确定，例如：

* 最大化 Youden index；
* 最大化 F1；
* 在 sensitivity ≥ 0.90 下获得最高 specificity。

[1]: https://huggingface.co/docs/timm/en/reference/optimizers?utm_source=chatgpt.com "Optimization"
