# 推荐模型结构

```python
import timm
import torch
import torch.nn as nn


class FundusClassifier(nn.Module):
    def __init__(
        self,
        model_name="convnext_tiny.fb_in22k_ft_in1k",
        meta_dim=16,
        dropout=0.2,
    ):
        super().__init__()

        # num_classes=0：
        # 移除原来的 ImageNet 分类层，直接输出全局池化后的特征
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        image_dim = self.backbone.num_features

        self.meta_encoder = nn.Sequential(
            nn.Linear(2, meta_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(image_dim + meta_dim),
            nn.Dropout(dropout),
            nn.Linear(image_dim + meta_dim, 1),
        )

    def forward(self, image, age, sex):
        image_feature = self.backbone(image)

        metadata = torch.stack([
            age.float(),
            sex.float(),
        ], dim=1)

        meta_feature = self.meta_encoder(metadata)

        feature = torch.cat([
            image_feature,
            meta_feature,
        ], dim=1)

        logit = self.classifier(feature).squeeze(1)
        return logit
```

timm 的 ConvNeXt 结构通常是：
```text
backbone.stem
backbone.stages[0]
backbone.stages[1]
backbone.stages[2]
backbone.stages[3]
backbone.head
```

## 分阶段解冻
* 前 2–3 个 epoch：冻结图像骨干，只训练年龄/性别分支和融合分类头。
* 接着解冻最后两个 stage。

先全部冻结：
```python
for param in model.backbone.parameters():
    param.requires_grad = False
```
然后解冻最后两个 stage：
```python
for stage in model.backbone.stages[-2:]:
    for param in stage.parameters():
        param.requires_grad = True

for param in model.backbone.head.parameters():
    param.requires_grad = True
```
最后全解冻：
```python
for param in model.backbone.parameters():
    param.requires_grad = True
```
需要注意：`create_model(..., num_classes=0)` 和 `backbone.num_features` 在不同模型间相对统一，但 `stages[-2:]` 属于具体模型结构。以后换成 EfficientNet、ResNet 或 ViT 时，部分解冻代码仍需单独适配。

例如更换模型通常只需要：
```text
model_name = "efficientnet_b2"
model_name = "resnet50"
model_name = "swin_tiny_patch4_window7_224"
model_name = "convnextv2_tiny.fcmae"
```