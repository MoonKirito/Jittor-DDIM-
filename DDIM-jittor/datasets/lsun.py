import os
import io
import pickle
import zipfile
import numpy as np
from PIL import Image
import lmdb

from jittor.dataset.dataset import Dataset

# 单个类别的 LSUN 数据集类
class LSUNClass(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        # 打开 LMDB 数据库
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

        # 生成或加载缓存 key 列表
        root_split = root.split("/")
        cache_file = os.path.join("/".join(root_split[:-1]), f"_cache_{root_split[-1]}")
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        # 从 LMDB 中读取图像数据
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = 0
        return img, target

    def __len__(self):
        return self.length

# 总的 LSUN 数据集封装类，可管理多个类别数据
class LSUN(Dataset):
    def __init__(self, root, classes="train", transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.classes = self._verify_classes(classes)

        # 为每个类别创建一个 LSUNClass 实例
        self.dbs = []
        for c in self.classes:
            self.dbs.append(
                LSUNClass(root=os.path.join(root, f"{c}_lmdb"), transform=transform)
            )

        # 用于索引分发
        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        # 验证并标准化类别参数
        categories = [
            "bedroom", "bridge", "church_outdoor", "classroom",
            "conference_room", "dining_room", "kitchen",
            "living_room", "restaurant", "tower"
        ]
        dset_opts = ["train", "val", "test"]

        if isinstance(classes, str):
            if classes == "test":
                return ["test"]
            else:
                return [c + "_" + classes for c in categories]

        # 如果是列表
        if not isinstance(classes, (list, tuple)):
            raise ValueError("classes 应该是字符串或字符串列表")

        # 验证列表元素是否合法
        result = []
        for c in classes:
            if not isinstance(c, str):
                raise ValueError("classes 中的元素应为字符串")
            c_short = c.split("_")
            category, dset_opt = "_".join(c_short[:-1]), c_short[-1]
            if category not in categories:
                raise ValueError(f"未知类别: {category}")
            if dset_opt not in dset_opts:
                raise ValueError(f"未知数据划分: {dset_opt}")
            result.append(c)
        return result

    def __getitem__(self, index):
        # 选择对应的子数据库
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub
        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length
