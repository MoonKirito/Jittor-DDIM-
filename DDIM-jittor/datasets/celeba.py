import os
import zipfile
import numpy as np
from PIL import Image
import pandas as pd

import jittor as jt
from jittor.dataset import Dataset
from datasets.utils import download_file_from_google_drive

class CelebA(Dataset):
    base_folder = "celeba"
    file_list = [
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False):
        super().__init__()
        self.root = root
        self.split = split
        self.target_type = target_type if isinstance(target_type, list) else [target_type]
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. Use download=True to fetch it.")

        if split == "train":
            split_id = 0
        elif split == "valid":
            split_id = 1
        elif split == "test":
            split_id = 2
        else:
            raise ValueError("Invalid split value. Must be one of 'train', 'valid', or 'test'.")

        base_path = os.path.join(self.root, self.base_folder)
        self.splits = pd.read_csv(os.path.join(base_path, "list_eval_partition.txt"), sep='\s+', header=None, index_col=0)
        self.identity = pd.read_csv(os.path.join(base_path, "identity_CelebA.txt"), sep='\s+', header=None, index_col=0)
        self.bbox = pd.read_csv(os.path.join(base_path, "list_bbox_celeba.txt"), sep='\s+', header=1, index_col=0)
        self.landmarks = pd.read_csv(os.path.join(base_path, "list_landmarks_align_celeba.txt"), sep='\s+', header=1, index_col=0)
        self.attr = pd.read_csv(os.path.join(base_path, "list_attr_celeba.txt"), sep='\s+', header=1, index_col=0)
        self.attr = (self.attr + 1) // 2  # 将 {-1, 1} 映射为 {0, 1}

        mask = (self.splits[1] == split_id)
        self.filenames = self.splits[mask].index.values
        self.identity = self.identity.loc[self.filenames].values
        self.bbox = self.bbox.loc[self.filenames].values
        self.landmarks = self.landmarks.loc[self.filenames].values
        self.attr = self.attr.loc[self.filenames].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filenames[index])
        img = Image.open(img_path).convert("RGB")

        # 构建目标属性
        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index])
            elif t == "identity":
                target.append(self.identity[index][0])
            elif t == "bbox":
                target.append(self.bbox[index])
            elif t == "landmarks":
                target.append(self.landmarks.iloc[index].values)
            else:
                raise ValueError(f"Unknown target type: {t}")
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        for _, md5, filename in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        if self._check_integrity():
            print("CelebA files already downloaded and verified.")
            return

        for file_id, md5, filename in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        zip_path = os.path.join(self.root, self.base_folder, "img_align_celeba.zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.root, self.base_folder))
