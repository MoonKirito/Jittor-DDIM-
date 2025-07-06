import os
import time
import pynvml

class JTLogger:
    def __init__(self, log_dir, filename="stdout.txt", gpu_id=0):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.start_time = time.time()
        self.gpu_id = gpu_id
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化 GPU 监控工具
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            self.gpu_handle = None
            print(f"[Warning] GPU monitor init failed: {e}")

        # 写入日志头
        with open(self.filepath, mode="w") as f:
            f.write("===== Jittor Training Log =====\n")
            f.write(f"Start Time: {time.ctime(self.start_time)}\n\n")

    def get_gpu_info(self):
        """获取当前 GPU 显存占用 (MB) 和利用率 (%)"""
        if self.gpu_handle is None:
            return None, None
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            return mem_info.used // (1024 * 1024), util.gpu
        except:
            return None, None

    def add_scalar(self, tag, value, step, epoch=None):
        """写入标量指标及 GPU 信息"""
        elapsed = time.time() - self.start_time
        mem_used, gpu_util = self.get_gpu_info()

        # 构造日志行
        epoch_info = f"[Epoch {epoch}] " if epoch is not None else ""
        gpu_info = (
            f" | GPU Mem: {mem_used}MB | Util: {gpu_util}%"
            if mem_used is not None else ""
        )
        log_line = f"{epoch_info}[{tag}] Step {step} | Value: {value:.6f} | Time: {elapsed:.2f}s{gpu_info}\n"

        # 写入文件
        with open(self.filepath, mode="a") as f:
            f.write(log_line)
