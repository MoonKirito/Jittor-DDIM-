import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = './train/logs/500epoch/stdout.txt'  # <-- 修改为你的实际路径

# 初始化数据容器
steps = []
losses = []
step_times = []

# 更新后的正则表达式，适配格式：Epoch 0 | Step 1 | Loss: 3300.709473 | Step Time: 2.62s
log_pattern = re.compile(
    r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Step Time:\s+([\d\.]+)s"
)

# 读取并解析日志
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

# 检查提取结果
if not steps:
    print("❌ 未能从日志中提取 Step/Loss/Step Time，请确认日志格式是否匹配。")
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
