import os
import lmdb
from PIL import Image
from io import BytesIO

import jittor as jt
from jittor.dataset.dataset import Dataset

# FFHQ 数据集类，使用 LMDB 格式加载图像
class FFHQ(Dataset):
    def __init__(self, path, transform, resolution=8):
        super().__init__()
        # 打开 LMDB 数据库
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("无法打开 FFHQ 的 LMDB 数据集: {}".format(path))

        # 获取数据集样本数量
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(b"length").decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 获取对应索引的图像数据（按照分辨率和填充后的编号组成键）
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        # 使用 PIL 从字节流中加载图像
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert("RGB")

        if self.transform:
            img = self.transform(img)

        target = 0  # FFHQ 不提供具体标签，用 0 占位
        return img, target
