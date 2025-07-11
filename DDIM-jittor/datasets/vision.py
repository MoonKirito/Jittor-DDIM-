import os
import jittor as jt
from jittor.dataset import Dataset

# 视觉数据集基类，继承自 Jittor 的 Dataset
class VisionDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        # 展开用户路径（如 ~/xxx）
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("只能传入 transforms 或 transform/target_transform，不能同时传入两者")

        # 为向后兼容保留 transform 和 target_transform
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError  # 子类需实现该方法

    def __len__(self):
        raise NotImplementedError  # 子类需实现该方法

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["数据集大小: {}".format(self.__len__())]
        if self.root is not None:
            body.append("数据根目录: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform, "图像变换: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "目标变换: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""  # 子类可扩展显示内容


# 用于封装 transform + target_transform
class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return '\n'.join(body)
