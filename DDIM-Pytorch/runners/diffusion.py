import os
import logging
import time
import glob
import imageio
import subprocess
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from cleanfid import fid
import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
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
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = config.tb_logger
        dataset, test_dataset = get_dataset(args, config)

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config).to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(config, model.parameters())

        ema_helper = None
        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)

        start_epoch, step = 0, 0

        if args.resume_training:
            states = torch.load(os.path.join(args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resumed training from epoch {start_epoch}, step {step}")

        for epoch in range(start_epoch, config.training.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                step_start = time.time()
                step += 1

                model.train()
                x = x.to(self.device)
                x = data_transform(config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                n = x.size(0)
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, t, e, b)
                tb_logger.add_scalar("loss", loss, global_step=step)

                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()

                if config.model.ema:
                    ema_helper.update(model)

                step_time = time.time() - step_start
                logging.info(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.6f} | Step Time: {step_time:.2f}s"
                )
                if step % config.training.snapshot_freq == 0 or step == 1:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(args.log_path, "ckpt.pth"))
                    logging.info(f"Saved model checkpoint at step {step}")

        final_states = [model.state_dict(), optimizer.state_dict(), epoch + 1, step]
        if config.model.ema:
            final_states.append(ema_helper.state_dict())

        torch.save(final_states, os.path.join(args.log_path, "ckpt_final.pth"))
        logging.info(f"Saved final model checkpoint at epoch {epoch + 1}, step {step}")

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt_final.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.gif:
            self.sample_gif(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"Starting from image {img_id}")
        total_n_samples = 218
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        total_time = 0
        total_generated = 0

        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size

                # 采样开始计时
                start_time = time.time()

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                # 采样结束
                end_time = time.time()
                total_time += (end_time - start_time)
                total_generated += n

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

        avg_time = total_time / total_generated
        print(f"\n平均每张图像采样时间: {avg_time:.4f} 秒，共生成 {total_generated} 张\n")

    def sample_gif(self, model):
        config = self.config
        device = self.device

        n = 128
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = list(range(0, self.num_timesteps, skip))
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                        ** 2
                )
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
                seq = (
                        np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            xs, _ = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        xs = [inverse_data_transform(config, x_) for x_ in xs]

        group1 = [x[:64] for x in xs]
        group2 = [x[64:] for x in xs]

        gif_path_1 = os.path.join(self.args.image_folder, "denoise_group1.gif")
        gif_path_2 = os.path.join(self.args.image_folder, "denoise_group2.gif")

        frames_1 = []
        frames_2 = []

        for t in range(len(seq)):
            grid1 = tvu.make_grid(group1[t], nrow=8, padding=2)
            grid2 = tvu.make_grid(group2[t], nrow=8, padding=2)
            frame1 = (grid1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame2 = (grid2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames_1.append(frame1)
            frames_2.append(frame2)

        imageio.mimsave(gif_path_1, frames_1, fps=5)
        imageio.mimsave(gif_path_2, frames_2, fps=5)

        print(f" GIF 已保存至:\n- {gif_path_1}\n- {gif_path_2}")

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                    torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i: i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        args, config = self.args, self.config
        dir1 = "./train/image_samples/t_50_eta_0/"
        dir2 = "./train/datasets/cifar10/cifar-10-batches-png/"
        score = fid.compute_fid(dir1, dir2)
        print("FID score:", score)



