最后训练的机器不是本机，但你可以使用本机的gpu进行测试

目标主机信息：
GPU：4×V100 32G
python环境:
    python: 3.9.7
    torch: 2.6.0+cu126
    torchvision: 0.21.0+cu126
    timm: 1.0.26
    webdataset: 0.2.100
    nvidia-dali-cuda120: 1.53.0

本机有一个名为ikang的conda环境，与目标主机的环境基本一致，你可以使用它来测试