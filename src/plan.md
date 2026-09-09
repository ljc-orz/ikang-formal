
模型架构见[model_example](./model/model_example.md)

默认数据增强见[transforms_example](./data/transforms_example.md)

训练策略见[config_example](./config/config_example.md)

先实现V1版本，采用上述的做法，但是全程只解冻模型的最后两个stage和外加的预测头。不做平衡采样。

模型最后将在一张32G的V100上训练，尽量选择最好的混合精度。训练集、内部测试集和外部测试集一共10万张图片。请尽量一个模块一个文件，不要有太复杂的引用关系，方便后续修改。

测试程序要支持分别对一个人两个眼睛预测然后取平均。

要有方便的启动脚本，可以自由选择训练哪些指标。

配有说明文档（写在本项目docs目录下）。

你只需要实现V1代码，然后做smoke test。