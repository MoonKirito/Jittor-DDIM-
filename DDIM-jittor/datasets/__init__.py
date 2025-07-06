import os
import numpy as np
from PIL import Image
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from jittor.dataset.cifar import CIFAR10
from jittor.transform import Resize, ToTensor, ImageNormalize, RandomHorizontalFlip, Compose

from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN

# 自定义 Crop 类用于图像裁剪
class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        # img 是 PIL 图像，裁剪区域定义为 (左，上，右，下)
        return img.crop((self.x1, self.y1, self.x2, self.y2))

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

# 获取数据集函数
def get_dataset(args, config):
    # 构建训练和测试用 transform
    if config.data.random_flip:
        train_transform = Compose([
            Resize(config.data.image_size),
            RandomHorizontalFlip(),
            ToTensor()
        ])
        test_transform = Compose([
            Resize(config.data.image_size),
            ToTensor()
        ])
    else:
        train_transform = test_transform = Compose([
            Resize(config.data.image_size),
            ToTensor()
        ])

    # 加载 CIFAR10 数据集
    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            train=True,
            transform=train_transform,
            download=True,
            root=os.path.join(args.exp, "datasets/cifar10")
        )
        test_dataset = CIFAR10(
            train=False,
            transform=test_transform,
            download=True,
            root=os.path.join(args.exp, "datasets/cifar10_test")
        )

    elif config.data.dataset == "CELEBA":
        cx, cy = 89, 121 # 先定义中心点为 (89, 121)
        x1, x2 = cy - 64, cy + 64
        y1, y2 = cx - 64, cx + 64
        crop = Crop(x1, x2, y1, y2)

        train_transform = Compose([crop, Resize(config.data.image_size), ToTensor()])
        if config.data.random_flip:
            train_transform.insert(2, RandomHorizontalFlip())

        test_transform = Compose([crop, Resize(config.data.image_size), ToTensor()])

        dataset = CelebA(root=os.path.join(args.exp, "datasets/celeba"), split="train", transform=train_transform, download=True)
        test_dataset = CelebA(root=os.path.join(args.exp, "datasets/celeba"), split="test", transform=test_transform, download=True)

    elif config.data.dataset == "LSUN":
        train_folder = f"{config.data.category}_train"
        val_folder = f"{config.data.category}_val"
        lsun_transform = Compose([Resize(config.data.image_size), ToTensor()])
        if config.data.random_flip:
            lsun_transform.insert(2, RandomHorizontalFlip())

        dataset = LSUN(root=os.path.join(args.exp, "datasets/lsun"), classes=[train_folder], transform=lsun_transform)
        test_dataset = LSUN(root=os.path.join(args.exp, "datasets/lsun"), classes=[val_folder], transform=lsun_transform)

    elif config.data.dataset == "FFHQ":
        ffhq_transform = [ToTensor()]
        if config.data.random_flip:
            ffhq_transform.insert(0, RandomHorizontalFlip())

        dataset = FFHQ(path=os.path.join(args.exp, "datasets/FFHQ"), transform=ffhq_transform, resolution=config.data.image_size)

        # 将 FFHQ 拆分成 train/test
        num_items = len(dataset)
        indices = list(range(num_items))
        np.random.seed(2019)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[:int(0.9*num_items)], indices[int(0.9*num_items):]

        class DatasetSubset(Dataset):
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = indices

            def __getitem__(self, index):
                return self.base[self.indices[index]]

            def __len__(self):
                return len(self.indices)

        dataset = DatasetSubset(dataset, train_indices)
        test_dataset = DatasetSubset(dataset, test_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset

def logit_transform(image, lam=1e-6): # 将图像从 [0,1] 区间映射到 logit 空间，避免出现 log(0)

    image = lam + (1 - 2 * lam) * image
    return jt.log(image) - jt.log(1 - image)


def data_transform(config, X): # 对原始输入图像 X 进行预处理转换
    if config.data.uniform_dequantization:
        # 均匀去量化：引入随机性，提高鲁棒性
        X = X / 256.0 * 255.0 + jt.rand(X.shape) / 256.0

    if config.data.gaussian_dequantization:
        # 高斯去量化：加噪声模拟采样误差
        X = X + jt.randn(X.shape) * 0.01

    if config.data.rescaled:
        # 将图像从 [0,1] 变换到 [-1,1]
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        # 对图像做 logit 变换
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        # 减去图像均值，实现归一化
        image_mean = jt.array(config.image_mean).reshape((1, -1, 1, 1))
        return X - image_mean

    return X


def inverse_data_transform(config, X): # 对模型输出图像进行反变换，恢复为可视化的图像数据
    if hasattr(config, "image_mean"):
        mean = jt.array(config.image_mean).reshape((1, -1, 1, 1))  # 明确对齐 shape
        X = X + mean

    if config.data.logit_transform:
        # 将 logit 输出还原为 sigmoid 范围
        X = nn.sigmoid(X)
    elif config.data.rescaled:
        # [-1,1] 映射回 [0,1]
        X = (X + 1.0) / 2.0

    # 保证值在 [0,1] 范围内
    return jt.clamp(X, 0.0, 1.0)