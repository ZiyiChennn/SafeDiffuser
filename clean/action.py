import torch
import os
import numpy as np
from os.path import join
# 假设你的文件是 'my_tensor.pt'






script_dir = os.path.dirname(os.path.abspath(__file__))

# 假设文件位于脚本目录下的 'data' 文件夹中

mode = 5 # Change this to 1-7 as needed
modes = {
    1: "actions_env.pt",
    2: "actions.pt",
    3: "observations_env.pt",
    4: "observations.pt",
    5: "safe1_all.pt",
    6: "safe2_all.pt",
    7: "scores.pt",
}
file_path = os.path.join(script_dir,modes.get(mode, "default"))

#file_path = os.path.join(script_dir, 'action.pt')
my_tensor = torch.load(file_path)
action_flat = my_tensor.reshape(-1, 2)   # shape = [3840, 2]

# 第0位的最小和最大
min0 = action_flat[:, 0].min()
max0 = action_flat[:, 0].max()

# 第1位的最小和最大
min1 = action_flat[:, 1].min()
max1 = action_flat[:, 1].max()

#print("第0位: min =", min0.item(), "max =", max0.item())
#print("第1位: min =", min1.item(), "max =", max1.item())
print(my_tensor)
#print(my_tensor.shape)
