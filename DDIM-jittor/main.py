import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import numpy as np
import jittor as jt
from functions.log import JTLogger
from runners.diffusion import Diffusion

jt.set_global_seed(1234)  # 设置 Jittor 的全局随机种子
jt.flags.use_cuda = 1
# 解析命令行参数和配置

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--exp", type=str, default="exp", help="实验目录")
    parser.add_argument("--doc", type=str, required=True, help="日志文件夹名")
    parser.add_argument("--comment", type=str, default="", help="实验备注")
    parser.add_argument("--verbose", type=str, default="info", help="日志级别")
    parser.add_argument("--test", action="store_true", help="是否测试模型")
    parser.add_argument("--sample", action="store_true", help="是否进行采样")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--resume_training", action="store_true", help="是否恢复训练")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="输出图像保存文件夹")
    parser.add_argument("--ni", action="store_true", help="无交互模式")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized", help="采样类型")
    parser.add_argument("--skip_type", type=str, default="uniform", help="跳跃类型")
    parser.add_argument("--timesteps", type=int, default=1000, help="采样步数")
    parser.add_argument("--eta", type=float, default=0.0, help="采样噪声系数")
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # 读取 YAML 配置
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = args.ni or input("日志文件夹已存在，是否覆盖？(Y/N) ").upper() == "Y"
                if overwrite:
                    shutil.rmtree(args.log_path)
                else:
                    print("终止程序。")
                    sys.exit(0)
            os.makedirs(args.log_path, exist_ok=True)
            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(config, f)

        # 添加日志记录器路径和初始化
        tb_path = os.path.join(args.exp, "tensorboard", args.doc)
        os.makedirs(tb_path, exist_ok=True)
        new_config.tb_logger = JTLogger(tb_path)  # 使用自定义的 logger

    else:
        if args.sample:
            args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
            if os.path.exists(args.image_folder) and not (args.fid or args.interpolation):
                overwrite = args.ni or input(f"图像文件夹 {args.image_folder} 已存在，是否覆盖？(Y/N) ").upper() == "Y"
                if overwrite:
                    shutil.rmtree(args.image_folder)
                else:
                    print("终止程序。")
                    sys.exit(0)
            os.makedirs(args.image_folder, exist_ok=True)

    # 日志设置
    level = getattr(logging, args.verbose.upper(), logging.INFO)
    logging.basicConfig(level=level,
                        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(args.log_path, "stdout.txt")) if not args.sample else logging.StreamHandler()
                        ])

    device = "cuda" if jt.has_cuda else "cpu"
    logging.info("使用设备: {}".format(device))
    new_config.device = device

    # 设置随机种子
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("日志路径: {}".format(args.log_path))
    logging.info("实验 ID: {}".format(os.getpid()))
    logging.info("备注信息: {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
