import os
import argparse
import torch
import imageio
import numpy as np
import torchvision.utils as tvu
from tqdm import tqdm
from models.diffusion import Model
from functions.ckpt_util import get_ckpt_path
from models.ema import EMAHelper
from functions.denoising import ddpm_steps
from datasets import inverse_data_transform
import yaml
from easydict import EasyDict as edict


def get_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)


def save_gif(sequence, output_path):
    imgs = [((img + 1.0) / 2.0).clamp(0, 1) for img in sequence]  # [-1,1] → [0,1]
    imgs = [img.cpu().numpy().transpose(1, 2, 0) * 255 for img in imgs]
    imgs = [img.astype(np.uint8) for img in imgs]
    imageio.mimsave(output_path, imgs, duration=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/cifar10.yml')
    parser.add_argument('--exp', default='train')
    parser.add_argument('--doc', default='epoch2000')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--ni', action='store_true')
    parser.add_argument('--sample_type', default='ddpm_noisy')
    parser.add_argument('--skip_type', default='uniform')
    parser.add_argument('--output_dir', default='./train/image_samples/gif_output')
    args = parser.parse_args()

    # ===== 设置路径 =====
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    args.image_folder = args.output_dir
    os.makedirs(args.image_folder, exist_ok=True)

    # ===== 加载配置 =====
    config = get_config(args.config)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.sampling = edict({
        "batch_size": 128,
    })

    # ===== 构建模型 =====
    model = Model(config).to(config.device)
    model = torch.nn.DataParallel(model)

    # ===== 加载 checkpoint =====
    ckpt = os.path.join(args.log_path, "ckpt_final.pth")
    states = torch.load(ckpt, map_location=config.device)
    model.load_state_dict(states[0], strict=True)

    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)

    model.eval()

    # ===== 开始采样 =====
    x = torch.randn(
        config.sampling.batch_size,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
        device=config.device,
    )

    skip = args.timesteps // 100
    seq = list(range(0, args.timesteps, skip))

    print(f"Sampling {config.sampling.batch_size} images, saving every {skip} steps...")

    all_steps = []
    with torch.no_grad():
        x_seq = ddpm_steps(x, seq, model, model.module.betas)
        for t, xt in enumerate(x_seq):
            xt = inverse_data_transform(config, xt)
            grid = tvu.make_grid(xt, nrow=16, padding=2, pad_value=1)
            tvu.save_image(grid, f"{args.image_folder}/step_{t:03d}.png")
            all_steps.append(grid)

    # ===== 保存为 GIF =====
    gif_path = os.path.join(args.output_dir, "sample_sequence.gif")
    print(f"Saving gif to {gif_path}")
    save_gif(all_steps, gif_path)


if __name__ == "__main__":
    main()
