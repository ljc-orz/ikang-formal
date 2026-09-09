# 推荐图像增强

```python
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = ...
IMAGE_PIXEL_MEAN = ...
IMAGE_PIXEL_STD = ...

train_transform = T.Compose(
    [
        # 保留大部分视野，仅模拟轻微取景差异
        T.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        # 对全身指标预测通常可以使用：左右眼镜像后结构仍合理
        T.RandomHorizontalFlip(p=0.5),
        # 模拟拍摄角度、位置和倍率差异
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
        # 只做温和颜色增强，避免破坏贫血、血红蛋白等颜色线索
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
        # 少量模拟轻微失焦
        T.RandomApply(
            [
                T.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 0.8),
                )
            ],
            p=0.1,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGE_PIXEL_MEAN,
            std=IMAGE_PIXEL_STD,
        ),
    ]
)

eval_transform = T.Compose(
    [
        T.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGE_PIXEL_MEAN,
            std=IMAGE_PIXEL_STD,
        ),
    ]
)
```