import os
import logging
import time
import glob
import pickle
import numpy as np
import tqdm
import subprocess
import jittor as jt
import jittor.dataset as data
import imageio
from models.diffusion import Model  # 自定义模型模块
from models.ema import EMAHelper  # 自定义 EMA 模块
from functions import get_optimizer  # 优化器配置函数
from functions.losses import loss_registry  # 损失函数注册表
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from PIL import Image


def torch2hwcuint8(x, clip=False):  # 将 Jittor 图像张量从 [-1, 1] 范围转换到 [0, 1] 范围
    if clip:
        x = jt.clamp(x, -1.0, 1.0)  # 将数值限制在 [-1, 1]
    x = (x + 1.0) / 2.0  # 映射到 [0, 1] 区间
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):  # 生成 beta 序列，用于扩散模型中的前向噪声添加

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)  # sigmoid 函数，用于平滑增长调度

    if beta_schedule == "quad":
        # β 随时间呈平方增长（根号均匀插值再平方）
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        # 线性均匀插值
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        # 常数 beta
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        # 近似 JSD 的形式：beta = 1/T, 1/(T-1), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        # Sigmoid 平滑过渡的 beta 增长曲线（非线性）
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        # 若未实现该策略，抛出异常
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def make_grid_jittor(tensor, nrow=8, padding=2):
    if isinstance(tensor, jt.Var):
        tensor = tensor.numpy()
    B, C, H, W = tensor.shape
    ncol = (B + nrow - 1) // nrow
    grid = np.zeros((C,
                     ncol * H + padding * (ncol - 1),
                     nrow * W + padding * (nrow - 1)),
                    dtype=np.float32)
    for idx in range(B):
        row = idx // nrow
        col = idx % nrow
        top = row * (H + padding)
        left = col * (W + padding)
        grid[:, top:top + H, left:left + W] = tensor[idx]
    return grid  # (C, H, W)


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:  # 设置默认设备为 GPU（如可用）否则为 CPU
            device = "cuda" if jt.has_cuda else "cpu"
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(  # 调用 beta 调度函数，生成长度为 T 的 beta 序列
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = jt.array(betas, dtype=jt.float32)  # 转为 Jittor 张量并存储
        self.num_timesteps = betas.shape[0]  # T 的长度
        alphas = 1.0 - betas  # 计算 alpha = 1 - beta，每一步的保留率
        alphas_cumprod = jt.cumprod(alphas, dim=0)  # 计算 alpha 累积乘积（用于公式 q(x_t | x_0) 中）
        alphas_cumprod_prev = jt.concat(  # 计算前一时间步的 alpha 累积乘积（用于后验方差）
            [jt.ones((1,), dtype=jt.float32), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (  # 计算后验方差
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":  # 设置 log 方差类型，根据配置中模型类型选择
            # 固定大方差，使用 beta 本身作为 logvar
            self.logvar = jt.log(betas)
        elif self.model_var_type == "fixedsmall":
            # 使用后验方差作为 logvar，并防止 log(0)
            self.logvar = jt.log(jt.maximum(posterior_variance, 1e-20)) \
 \
                    def train(self):
        jt.flags.use_cuda = 1
        jittor_vis.start()
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # 加载训练集和测试集
        dataset, test_dataset = get_dataset(args, config)
        train_loader = dataset.set_attrs(
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )

        # 构建模型
        model = Model(config)

        # 构建优化器
        optimizer = get_optimizer(config, model.parameters())

        # EMA 权重维护器
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # 断点恢复
        start_epoch, step = 0, 0
        if args.resume_training:
            states = jt.load(os.path.join(args.log_path, "ckpt.pkl"))
            model.load_parameters(states[0])
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if config.model.ema:
                ema_helper.load_state_dict(states[4])

        best_loss = float('inf')
        # 开始训练
        for epoch in range(start_epoch, config.training.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                step_start = time.time()

                n = x.shape[0]
                step += 1
                model.train()

                # 数据 + 预处理
                x = x.stop_grad().to(self.device)
                x = data_transform(config, x)

                # 噪声 & 时间步
                e = jt.randn_like(x)
                b = self.betas
                t = jt.randint(0, self.num_timesteps, shape=[n // 2 + 1])
                t = jt.concat([t, self.num_timesteps - t - 1])[:n]

                # 前向 + 损失
                loss = loss_registry[config.model.type](model, x, t, e, b)

                # 反向传播 + 更新
                optimizer.step(loss)

                # EMA 更新
                if config.model.ema:
                    ema_helper.update(model)

                # 计算 step 总时长
                step_time = time.time() - step_start

                # 日志记录
                tb_logger.add_scalar("loss", loss.item(), step, epoch)
                tb_logger.add_scalar("step_time", step_time, step, epoch)

                logging.info(
                    f"[Epoch {epoch}] Step: {step}, "
                    f"Loss: {loss.item():.6f}, Step Time: {step_time:.3f}s"
                )

                # 每 snapshot_freq 步保存一次模型
                if step % config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if config.model.ema:
                        states.append(ema_helper.state_dict())

                    jt.save(states, os.path.join(args.log_path, f"ckpt_{step}.pkl"))
                    jt.save(states, os.path.join(args.log_path, "ckpt.pkl"))
                    logging.info(f" 已保存模型 checkpoint at step {step}")

        #  训练结束后保存最终模型
        final_states = [model.state_dict(), optimizer.state_dict(), config.training.n_epochs, step]
        if config.model.ema:
            final_states.append(ema_helper.state_dict())

        jt.save(final_states, os.path.join(args.log_path, "ckpt_final.pkl"))
        logging.info(f"训练完成，已保存最终模型至 ckpt_final.pkl")

    def sample(self):
        # 创建模型
        model = Model(self.config)
        # 情况一：不使用预训练模型，加载训练过程中保存的模型
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = jt.load(
                    os.path.join(self.args.log_path, "ckpt_700000.pkl")
                )
            else:
                states = jt.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    )
                )
            model.load_parameters(states[0])  # 加载参数
            # 如果启用 EMA，就还原 EMA 参数
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)  # 将 EMA 参数应用到当前模型
            else:
                ema_helper = None
        else:
            # 情况二：使用预训练模型（默认从原始作者处下载）
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError("Unsupported dataset")
            ckpt = r"C:\Users\Kirit\.cache\diffusion_models_converted\ema_diffusion_cifar10_model\model-790000.pkl"  # 你手动转换后的 pkl 路径
            print("Loading checkpoint from local pkl:", ckpt)
            model.load_parameters(jt.load(ckpt))
        model.eval()  # 设置模型为评估模式
        # 根据指定的采样任务调用对应方法
        if self.args.fid:
            self.sample_fid(model)
        elif self.args.gif:
            self.sample_gif(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedure not defined")

    def sample_fid(self, model):
        config = self.config
        # 当前已有图像数量（用于接着生成）
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"Starting from image {img_id}")

        # 总共需要生成的图像数
        total_n_samples = 128
        # 总共采样轮数 = 剩余需要生成图像数 / 每轮采样的图像数
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        total_time = 0
        total_generated = 0

        with jt.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size

                # 采样开始计时
                start_time = time.time()

                # 初始化随机噪声图像作为采样起点
                x = jt.randn((n, config.data.channels, config.data.image_size, config.data.image_size))

                # 进行采样（使用DDIM或DDPM等采样策略）
                x = self.sample_image(x, model)

                # 执行反数据归一化，将图像从[-1,1]还原为[0,1]
                x = inverse_data_transform(config, x)

                # 采样结束计时
                end_time = time.time()
                total_time += (end_time - start_time)
                total_generated += n

                # 保存生成的图像
                for i in range(n):
                    np_img = (x[i].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(np_img).save(os.path.join(self.args.image_folder, f"{img_id}.png"))
                    img_id += 1

        # 输出平均采样时间
        avg_time = total_time / total_generated
        print(f"\n 平均每张图像采样时间: {avg_time:.4f} 秒，共生成 {total_generated} 张\n")

    def sample_gif(self, model):
        config = self.config

        n = 128
        x = jt.randn((n, config.data.channels, config.data.image_size, config.data.image_size))

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(0, self.num_timesteps, skip))
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            xs, _ = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(0, self.num_timesteps, skip))
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            xs, _ = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        xs = [inverse_data_transform(config, x_) for x_ in xs]

        group1 = [x_[:64] for x_ in xs]
        group2 = [x_[64:] for x_ in xs]

        gif_path_1 = os.path.join(self.args.image_folder, "denoise_group1.gif")
        gif_path_2 = os.path.join(self.args.image_folder, "denoise_group2.gif")

        frames_1 = []
        frames_2 = []

        for t in range(len(seq)):
            # grid 拼图 + 转为 uint8
            grid1 = make_grid_jittor(group1[t], nrow=8, padding=2)
            grid2 = make_grid_jittor(group2[t], nrow=8, padding=2)

            frame1 = (grid1.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            frame2 = (grid2.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

            frames_1.append(frame1)
            frames_2.append(frame2)

        imageio.mimsave(gif_path_1, frames_1, fps=5)
        imageio.mimsave(gif_path_2, frames_2, fps=5)

        print(f" GIF 已保存至:\n- {gif_path_1}\n- {gif_path_2}")

    def sample_sequence(self, model):
        config = self.config
        # 初始化随机噪声图像（8 张），尺寸为 image_size，通道数为 data.channels
        x = jt.randn(
            (8, config.data.channels, config.data.image_size, config.data.image_size),
            dtype=jt.float32
        )
        # 关闭梯度计算，执行采样过程（获得每一步 x0 而不是 x_{t-1}）
        with jt.no_grad():
            _, x_seq = self.sample_image(x, model, last=False)
        # 反变换，将 [-1, 1] 区间的张量还原为图像
        x_seq = [inverse_data_transform(config, y) for y in x_seq]
        # 遍历每一时间步和每一张图片，将其保存为 PNG 图像
        for i in range(len(x_seq)):
            for j in range(x_seq[i].shape[0]):
                np_img = np.clip((x_seq[i][j].numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
                Image.fromarray(np_img).save(os.path.join(self.args.image_folder, f"{j}_{i}.png"))

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            # 球面线性插值 (Spherical Linear Interpolation)
            dot = (z1 * z2).sum()
            norm1 = jt.norm(z1)
            norm2 = jt.norm(z2)
            theta = jt.acos(dot / (norm1 * norm2))
            return (
                    jt.sin((1 - alpha) * theta) / jt.sin(theta) * z1
                    + jt.sin(alpha * theta) / jt.sin(theta) * z2
            )

        # 生成两组随机噪声 z1 和 z2
        z1 = jt.randn((1, config.data.channels, config.data.image_size, config.data.image_size))
        z2 = jt.randn((1, config.data.channels, config.data.image_size, config.data.image_size))
        # 构建 alpha 插值因子 [0.0, 0.1, ..., 1.0]
        alpha = jt.linspace(0.0, 1.0, 11)
        z_list = [slerp(z1, z2, a) for a in alpha]
        # 拼接所有插值结果，作为模型输入
        x = jt.concat(z_list, dim=0)
        xs = []
        # 批量采样图像，每次最多采样8张
        with jt.no_grad():
            for i in range(0, x.shape[0], 8):
                xs.append(self.sample_image(x[i:i + 8], model))
        # 合并所有采样图像并执行反数据归一化
        x = jt.concat(xs, dim=0)
        x = inverse_data_transform(config, x)
        # 保存每张图片为PNG格式
        for i in range(x.shape[0]):
            np_img = np.clip((x[i].numpy().transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
            Image.fromarray(np_img).save(os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        # 根据采样类型选择采样过程
        if self.args.sample_type == "generalized":
            # 构建时间步序列 seq
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError("不支持的 skip_type")
            # 调用 generalized 采样方法
            from functions.denoising import generalized_steps
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs  # xs 是 (x_t 序列, x0 序列)
        elif self.args.sample_type == "ddpm_noisy":
            # 同样构建时间步序列 seq
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps) ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError("不支持的 skip_type")
            # 调用 DDPM 步骤
            from functions.denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError("不支持的 sample_type")
        # 如果只返回最终结果
        if last:
            x = x[0][-1]  # 取 xt 序列中的最终结果
        return x

    def test(self):
        args, config = self.args, self.config
        dir1 = "./train/image_samples/t_1000_eta_0/"
        dir2 = "./train/datasets/cifar10/cifar-10-batches-png/"
        score = fid.compute_fid(dir1, dir2)
        print("FID score:", score)

