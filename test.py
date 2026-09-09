from src.data import FundusWebDataset

dataset = FundusWebDataset(
    "example/webdataset",
    "train",
    "result_alt",
)

for image, age, sex, result in dataset:
    # image: torch.uint8，[3, H, W]
    # age: int
    # sex: MAN=0，WOMAN=1
    # result: 0/1，缺失时为 -1
    pass
