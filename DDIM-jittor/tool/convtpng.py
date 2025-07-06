import os
import pickle
import numpy as np
from PIL import Image

# 输入和输出目录
input_dir = './train/datasets/cifar10/cifar-10-batches-py'
output_dir = './train/datasets/cifar10/cifar-10-batches-png'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载 .pkl 文件的函数
def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# 处理单个批次的函数
def process_batch(file_name, batch_index):
    batch = unpickle(os.path.join(input_dir, file_name))
    data = batch[b'data']
    labels = batch[b'labels']

    for i in range(len(data)):
        img_flat = data[i]

        # 分离通道并 reshape 为 32x32
        r = img_flat[0:1024].reshape((32, 32))
        g = img_flat[1024:2048].reshape((32, 32))
        b = img_flat[2048:3072].reshape((32, 32))
        img = np.stack([r, g, b], axis=2)

        # 保存PNG图像
        img_pil = Image.fromarray(img)
        filename = f'{file_name}_{i:05d}.png'
        img_pil.save(os.path.join(output_dir, filename))

# 所有要处理的批次文件
batch_files = [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']

# 批量转换
for idx, file in enumerate(batch_files):
    print(f"Processing {file}...")
    process_batch(file, idx)

print("所有图像已保存到:", output_dir)
