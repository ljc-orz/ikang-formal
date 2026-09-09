# 眼底数据 WebDataset 组织

转换后按数据划分建立三个目录；每个目录是一组可独立扩展的 tar 分片，而不是把真实数据强行塞进一个超大 tar：

```text
OUTPUT/
├── dataset.json
├── skipped.jsonl                         # 仅在跳过缺图记录时存在
├── train/
│   ├── train-000000.tar
│   ├── train-000000.idx
│   └── ...
├── external_validation/
│   ├── external_validation-000000.tar
│   ├── external_validation-000000.idx
│   └── ...
└── internal_validation/
    ├── internal_validation-000000.tar
    ├── internal_validation-000000.idx
    └── ...
```

同一个患者使用同一个无点号 key，例如 `00000042`，在 tar 中组成一个 WebDataset sample：

```text
00000042.left.jpg       # image_path_1 的原始 JPEG 字节
00000042.right.jpg      # image_path_2 的原始 JPEG 字节
00000042.labels.npy     # int8[指标数]，顺序记录于 dataset.json
00000042.meta.json      # 原 Parquet 的非 result_* 字段及 source_row
```

标签只允许 `0`、`1` 或空值；空值编码为 `-1`。`dataset.json` 是机器可读的数据契约，记录标签顺序、各 split 数量、分片路径和 DALI 的 component extension。`.idx` 是 DALI `wds2idx` 生成的随机访问索引。tar 不压缩，适合 DALI mmap 和顺序读取。

## 生成

在项目根目录运行：

```bash
conda run -n AnaCP python data_prepare/build_webdataset.py \
  --parquet example/example.parquet \
  --image-root example/raw \
  --output example/webdataset \
  --max-samples-per-shard 2000 \
  --max-shard-size-gb 2
```

默认遇到缺图立即失败，且不会替换已有输出。确认允许丢弃整位患者时显式增加 `--missing-image skip`，跳过详情会写入 `skipped.jsonl`。重新生成已有目录需增加 `--overwrite`；旧目录只会在新数据完整生成后被替换。

两个上限任一达到就切新分片。对于 5 万行真实数据，建议保留默认的每片 2,000 位患者、约 2 GiB 上限；训练时可以在多个 tar 间 shuffle。验证集同样可以有多片，但始终与训练集物理隔离。

## NVIDIA DALI 读取

```python
from nvidia.dali import fn, pipeline_def, types


@pipeline_def(batch_size=32, num_threads=4, device_id=0)
def train_pipe(tars, indexes):
    left_raw, right_raw, labels_raw, metadata = fn.readers.webdataset(
        paths=tars,
        index_paths=indexes,
        ext=["left.jpg", "right.jpg", "labels.npy", "meta.json"],
        missing_component_behavior="error",
        random_shuffle=True,
        name="Reader",
    )
    left = fn.decoders.image(left_raw, device="mixed", output_type=types.RGB)
    right = fn.decoders.image(right_raw, device="mixed", output_type=types.RGB)
    labels = fn.decoders.numpy(labels_raw)
    return left, right, labels, metadata


tars = ["OUTPUT/train/train-000000.tar"]
indexes = ["OUTPUT/train/train-000000.idx"]
pipe = train_pipe(tars=tars, indexes=indexes)
pipe.build()
```

实际使用时将同一 split 下排序后的所有 `.tar` 与对应 `.idx` 传入，两个列表必须一一对应。`meta.json` 是 UTF-8 原始字节；训练主路径通常只取两张图和 `labels.npy`，评估或追溯时再解析 metadata。

普通 Python WebDataset 也可读取：

```python
import webdataset as wds

dataset = (
    wds.WebDataset("OUTPUT/train/train-{000000..000024}.tar", shardshuffle=True)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("left.jpg", "right.jpg", "labels.npy", "meta.json")
)
```
