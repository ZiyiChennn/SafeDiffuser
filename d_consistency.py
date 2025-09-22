import torch
import torch.nn as nn
import numpy as np
from diffuser.normalization import DatasetNormalizer, GaussianNormalizer, flatten
# 定义您提供的模型结构
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, dt=0.008):
        super().__init__()
        half_hidden_dim = hidden_dim
        self.dt = dt
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, half_hidden_dim),
            nn.ELU(),
            nn.Linear(half_hidden_dim, state_dim)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        ds_dt = self.net(x)
        # 根据模型forward函数，预测的是s_{i+1} = s_i + ds_dt
        return s + ds_dt 

# 定义模型参数
state_dim = 17
action_dim = 6

# 实例化模型
model = DynamicsModel(state_dim=state_dim, action_dim=action_dim)



ex_re_mes ={
    1:"walker/expert/model_ema_esemble_0_4.pth",
    2:"walker/replay/model_ema_esemble_0_18.pth",
    3:"walker/medium/model_ema_esemble_0_23.pth",
    4:"hopper/expert/model_ema_esemble_0_7.pth",
    5:"hopper/replay/model_ema_esemble_0_7.pth",
    6:"hopper/medium/model_ema_esemble_0_23.pth"
    } 
ex_re_me = ex_re_mes[2]


model_path = f'/root/SafeDiffuser/diffuser/dynamische_model/{ex_re_me}'

model.load_state_dict(torch.load(model_path))


model.eval()

# 将模型移动到可用的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch


names = {
    1: "diffuser",
    2: "GD",
    4: "invariance",
    5: "invariance_cf",
    6: "invariance_cpx",
    7: "invariance_cpx_cf",
    3: "Shield",


    8:  "diffuser_hopper", 
    9:  "GD_hopper",
    10: "Shield_hopper",
    11: "invariance_hopper",
    12: "invariance_hopper_cf",
    13: "invariance_hopper_cpx",
    14: "invariance_hopper_cpx_cf",
   
}
name = names[1] # Change this to 1-12 as needed

si_data = torch.load(f"/root/SafeDiffuser/logs/walker2d-medium-replay-v2/plans/H600_T20_d0.99/40/{name}/plan_observation.pt")  # [6496, 600, 17]
ai_data = torch.load(f'/root/SafeDiffuser/logs/walker2d-medium-replay-v2/plans/H600_T20_d0.99/40/{name}/plan_action.pt')  # [6496, 600, 6]


# 将数据移动到设备上
si_data = si_data.to(device)
ai_data = ai_data.to(device)
dataset = {
    'observations': si_data.cpu().numpy(),  # [6496, 600, 17]
    'actions': ai_data.cpu().numpy()       # [6496, 600, 6]
}
path_lengths = [600] * si_data.shape[0]
normalizer = DatasetNormalizer(dataset, 'GaussianNormalizer', path_lengths=path_lengths)

# 打印正则化统计量以调试
print("\n状态特征均值:", normalizer.normalizers['observations'].means)
print("状态特征标准差:", normalizer.normalizers['observations'].stds)
print("动作特征均值:", normalizer.normalizers['actions'].means)
print("动作特征标准差:", normalizer.normalizers['actions'].stds)

print("\n正在为每条轨迹计算并存储 d_values...")
all_d_values = []  # 存储所有轨迹的 d_values

with torch.no_grad():
    for i in range(si_data.shape[0]):
        si_trajectory = si_data[i].unsqueeze(0).to(device)  # [1, 600, 17]
        ai_trajectory = ai_data[i].unsqueeze(0).to(device)  # [1, 600, 6]
        
        # 正则化状态和动作
        si_flat = si_trajectory.cpu().numpy().reshape(-1, state_dim)  # [600, 17]
        ai_flat = ai_trajectory.cpu().numpy().reshape(-1, action_dim)  # [600, 6]
        si_norm = normalizer.normalize(si_flat, 'observations')  # [600, 17]
        ai_norm = normalizer.normalize(ai_flat, 'actions')      # [600, 6]
        
        # 转换回张量并恢复形状
        si_norm = torch.tensor(si_norm, dtype=torch.float32, device=device).reshape(1, 600, state_dim)
        ai_norm = torch.tensor(ai_norm, dtype=torch.float32, device=device).reshape(1, 600, action_dim)
        
        s_i = si_norm[:, :-1, :]  # [1, 599, 17]
        a_i = ai_norm[:, :-1, :]  # [1, 599, 6]

        s_i_plus_1_true = si_norm[:, 1:, :]  # [1, 599, 17]（正则化）


        s_i_plus_1_pred = model(s_i, a_i)  # [1, 599, 17]
        
        
        
        diff = s_i_plus_1_true - s_i_plus_1_pred  # [1, 599, 17]
        squared_norm = torch.sum(diff**2, dim=-1)  # [1, 599]
        d_values = 0.5 * squared_norm  # [1, 599]
        
        all_d_values.append(d_values.cpu().numpy())  # 存储 [1, 599]

# 转换为 NumPy 数组
all_d_values = np.array(all_d_values)  # [6496, 1, 599]
all_d_values = all_d_values.squeeze(1)  # [6496, 599]

print(f"\nall_d_values 形状: {all_d_values.shape}")

# --- 第二步: 计算每条轨迹的 mean, std, median ---
d_mean_array = np.mean(all_d_values, axis=1)  # [6496]
d_std_array = np.std(all_d_values, axis=1)    # [6496]
d_median_array = np.median(all_d_values, axis=1)  # [6496]

print(f"d_mean_array 形状: {d_mean_array.shape}")
print(f"d_std_array 形状: {d_std_array.shape}")
print(f"d_median_array 形状: {d_median_array.shape}")

# --- 第三步: 根据里程碑分段 ---
milestone_numbers = np.array([1.254654, 1.250145, 1.2515, 1.247525, 1.25258, 1.247683, 1.254855, 1.254187, 1.252626, 1.252102])
tolerance = 1e-6
full_array_obs = si_data.cpu().numpy()[:, 0, 0]  # [6496]
milestone_indices = []
start_search_index = 0

for i, milestone in enumerate(milestone_numbers):
    # 从上一个找到的索引之后开始搜索
    current_search_segment = full_array_obs[start_search_index:]
   
    
    # 在当前搜索段内查找所有匹配的索引
    indices = np.where(np.abs(current_search_segment - milestone) < tolerance)[0]
    
    if indices.size > 0:
        # 找到第一个匹配的索引
        first_match_in_segment = indices[0]
        # 转换回原始数组的全局索引
        global_index = start_search_index + first_match_in_segment
        milestone_indices.append(global_index)
        
        # 更新下一次搜索的起点为当前找到的索引之后
        start_search_index = global_index + 1
    else:
        print(f"错误: 在索引 {start_search_index} 之后找不到里程碑数字 {milestone}。")
        # 如果找不到，整个过程就无法继续
        exit()

# 确认找到了10个有效的索引
if len(milestone_indices) != 10:
    print(f"错误: 最终找到的里程碑索引数量不等于10。找到 {len(milestone_indices)} 个。")
    exit()

# 定义 10 个分段
segments = []
for i in range(len(milestone_indices) - 1):
    start_idx = milestone_indices[i]
    end_idx = milestone_indices[i + 1]
    segments.append((start_idx, end_idx))
segments.append((milestone_indices[-1], len(d_mean_array)))

print(f"\n找到 {len(segments)} 个分段。正在处理每个分段...")

# 打印每段批次个数
print("\n--- 每个段的批次个数 ---")
print("-----------------------")
for i, (start_idx, end_idx) in enumerate(segments):
    segment_length = end_idx - start_idx
    start_num = milestone_numbers[i] if i < len(milestone_numbers) else milestone_numbers[-1]
    end_num = milestone_numbers[i + 1] if i < len(milestone_numbers) - 1 else "结尾"
    print(f"第 {i+1} 段 ({start_num:.6f} 到 {end_num}): {segment_length} 个批次")

# --- 第四步: 计算每段的统计量平均值 ---
segment_means = []
segment_stds = []
segment_medians = []

for i, (start_idx, end_idx) in enumerate(segments):
    if end_idx <= start_idx:
        print(f"警告: 分段 {i+1} 为空或长度为负。跳过。")
        continue
    
    segment_d_mean = d_mean_array[start_idx:end_idx]
    segment_d_std = d_std_array[start_idx:end_idx]
    segment_d_median = d_median_array[start_idx:end_idx]
    
    mean_of_means = np.mean(segment_d_mean)
    mean_of_stds = np.mean(segment_d_std)
    mean_of_medians = np.mean(segment_d_median)
    
    segment_means.append(mean_of_means)
    segment_stds.append(mean_of_stds)
    segment_medians.append(mean_of_medians)

# 打印最终结果
print("\n--- 分段统计量平均值 ---")
print("-----------------------")
for i in range(len(segment_means)):
    print(f"分段 {i+1}:")
    print(f"  d_mean 的平均值: {segment_means[i]:.3f}")
    print(f"  d_std 的平均值: {segment_stds[i]:.3f}")
    print(f"  d_median 的平均值: {segment_medians[i]:.3f}")