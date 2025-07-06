import re
import matplotlib.pyplot as plt

# 替换为你的日志文件路径
log_file_path = './train/logs/500epoch/stdout.txt'

# 存储提取的数据
steps = []
losses = []
step_times = []

# 正则表达式匹配训练日志行
log_pattern = re.compile(
    r"Step:\s*(\d+),\s*Loss:\s*([\d\.]+),\s*Step Time:\s*([\d\.]+)s"
)

# 读取日志并提取数据
with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = log_pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            step_time = float(match.group(3))

            steps.append(step)
            losses.append(loss)
            step_times.append(step_time)

# 检查是否成功提取数据
if not steps:
    print("未能从日志中提取到任何 Step/Loss 数据，请确认日志格式正确。")
    exit()

# 绘制 Loss 曲线图
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o', label='Loss')
plt.title('Loss Curve per Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# 绘制 Step Time 折线图
plt.figure(figsize=(10, 5))
plt.plot(steps, step_times, marker='x', linestyle='--', color='orange', label='Step Time')
plt.title('Step Time per Step')
plt.xlabel('Step')
plt.ylabel('Step Time (seconds)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("step_time_curve.png")
plt.show()
