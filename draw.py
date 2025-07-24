import torch 
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==== 参数设置 ====
log_dir = "tabelle/walker/expert"
base_name = "walker__expert__invariance_1000_fn"
step = 0          # 0~9
obs_index = 0     # 0~15

# ==== 加载 observation ====
obs_path = os.path.join(log_dir, f"{base_name}.observation.pt")
observations = torch.load(obs_path)
print("Loaded observations shape:", observations.shape)

# ==== 加载 action ====
act_path = os.path.join(log_dir, f"{base_name}.action.pt")
actions = torch.load(act_path)
print("Loaded actions shape:", actions.shape)

# ==== 创建上下两个子图 ====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))  # 2行1列

# ==== 画 observation ====
ax1.axhline(y=1.3, color='red', linestyle='--', linewidth=1, label='y=1.3')
y_obs = observations[step, obs_index]  # shape: [1000]
ax1.plot(y_obs, label=f"obs[{obs_index}]")
ax1.set_title(f"Observation[{obs_index}] in {step}.Epsiode")
ax1.set_xlabel("Time step t")
ax1.set_ylabel("Observation Value")
ax1.grid(True)
ax1.legend()

# ==== 画 action ====
for act_index in range(3):
    y_act = actions[step, act_index]  # shape: [1000]
    ax2.plot(y_act, label=f"act[{act_index}]")

# 添加 y=1 和 y=-1 两条虚线
ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, label='y=1')
ax2.axhline(y=-1, color='red', linestyle='--', linewidth=1, label='y=-1')

ax2.set_title(f"Action in {step}.Epsiode")
ax2.set_xlabel("Time step t")
ax2.set_ylabel("Action Value")
ax2.grid(True)
ax2.legend()

# ==== 保存 ====
plt.tight_layout()
plt.savefig(f"{base_name}_{step}_obs_and_action.png")
print("Figure saved.")
