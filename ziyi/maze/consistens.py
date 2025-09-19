import torch
import torch.nn as nn
import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, GaussianNormalizer, flatten
# 定义您提供的模型结构
import torch
import torch.nn as nn

import torch
import torch.nn as nn

# explicit dynamic model
class Maze2dExpDynamicsModel(nn.Module):
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt

    def forward(self, s:torch.tensor, a:torch.tensor):
        """
        s [torch.tensor]: [batch, hor, obs_dim], obs_dim=4, [x,y,vx,vy], 
        a [torch.tensor]: [batch, hor, act_dim], act_dim=2, [fx,fy]
        """        
        # Euler integration
        v_next = s[:,:,2:4] + self.dt * a
        p_next = s[:,:,0:2] + self.dt * s[:,:,2:4]

        next_obs = torch.cat([p_next, v_next], dim=-1)
        return next_obs
    
class Maze2dNNDynamicsModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)  # Output is Δs (state difference)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        delta_s = self.net(x)
        return s + delta_s # Euler integration

# 定义模型参数
state_dim = 4
action_dim = 2

# 实例化模型
model = Maze2dNNDynamicsModel(state_dim=state_dim, action_dim=action_dim)



la_ums ={
    1: "large/model_ema_esemble_0_25.pth",
    2: "umaze/model_ema_esemble_0_13.pth"
    } 
la_um = la_ums[1]


model_path = f'/root/SafeDiffuser/diffuser/dynamische_model/{la_um}'

model.load_state_dict(torch.load(model_path))


model.eval()

# 将模型移动到可用的设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch


names = {
    1: "diffuser",
    2: "GD",
    3: "in_re",
    4: "in_re_cf",
    5: "in_re_na",
    6: "in_ti",
    7: "in_ti_cf",
    8: "shield",



}
name = names[3] # Change this to 1-12 as needed

si_data = torch.load(f"/root/SafeDiffuser/logs/maze2d-umaze-v1/plan/{name}/observations.pt")  
ai_data = torch.load(f'/root/SafeDiffuser/logs/maze2d-umaze-v1/plan/{name}/actions.pt')  

# 将数据移动到设备上
si_data = si_data.to(device)
ai_data = ai_data.to(device)
dataset = {
    'observations': si_data.cpu().numpy(),  
    'actions': ai_data.cpu().numpy()      
}
path_lengths = [128] * si_data.shape[0]
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
        si_trajectory = si_data[i].unsqueeze(0).to(device)  
        ai_trajectory = ai_data[i].unsqueeze(0).to(device)  
        
        # 正则化状态和动作
        si_flat = si_trajectory.cpu().numpy().reshape(-1, state_dim)  
        ai_flat = ai_trajectory.cpu().numpy().reshape(-1, action_dim) 
        si_norm = normalizer.normalize(si_flat, 'observations')  
        ai_norm = normalizer.normalize(ai_flat, 'actions')      
        
        # 转换回张量并恢复形状
        si_norm = torch.tensor(si_norm, dtype=torch.float32, device=device).reshape(1,128, state_dim)
        ai_norm = torch.tensor(ai_norm, dtype=torch.float32, device=device).reshape(1,128, action_dim)
        
        s_i = si_norm[:, :-1, :]  
        a_i = ai_norm[:, :-1, :]  

        s_i_plus_1_true = si_norm[:, 1:, :]  

        s_i_plus_1_pred = model(s_i, a_i)  
        
        
        diff = s_i_plus_1_true - s_i_plus_1_pred  
        squared_norm = torch.sum(diff**2, dim=-1)  
        d_values = 0.5 * squared_norm  
        
        all_d_values.append(d_values.cpu().numpy())  

# 转换为 NumPy 数组
all_d_values = np.array(all_d_values)  
all_d_values = all_d_values.squeeze(1)  

print(f"\nall_d_values 形状: {all_d_values.shape}")

# --- 第二步: 计算每条轨迹的 mean, std, median ---
d_mean_array = np.mean(all_d_values, axis=1)  
d_std_array = np.std(all_d_values, axis=1)    
d_median_array = np.median(all_d_values, axis=1)  

rounded_d_mean = np.round(d_mean_array, 3)
rounded_d_std = np.round(d_std_array, 3)
rounded_d_median = np.round(d_median_array, 3)


print(f"d_mean_array 形状: {rounded_d_mean}")
print(f"d_std_array 形状: {rounded_d_std}")
print(f"d_median_array 形状: {rounded_d_median}")






