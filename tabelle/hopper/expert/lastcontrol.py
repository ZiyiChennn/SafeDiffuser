import torch
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
obs_file_path = os.path.join(script_dir, "hopper__expert__invariance_1000_fn_observation.pt")

try:
    obs_data = torch.load(obs_file_path)
    
    # --- 修复逻辑：根据数据类型进行转换 ---
    if isinstance(obs_data, torch.Tensor):
        full_array = obs_data.numpy()
        print("已将 PyTorch 张量转换为 NumPy 数组。")
    elif isinstance(obs_data, np.ndarray):
        full_array = obs_data
        print("加载的数据已经是 NumPy 数组。")
    else:
        print(f"错误: 无法处理的数据类型: {type(obs_data)}")
        exit()

    print(f"观察数据形状: {full_array.shape}")
    print(f"数据类型: {full_array.dtype}")
    
    # 使用 NumPy 函数检查 NaN 和 Inf
    if np.isnan(full_array).any():
        print("\n诊断: 数组中存在 NaN (非数字) 值!")
        
    if np.isinf(full_array).any():
        print("诊断: 数组中存在 inf (无穷大) 值!")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {obs_file_path}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载或处理文件时发生错误: {e}")
    exit()

# 如果数组本身没有 NaN，那么问题可能在循环中
max_values = []

# 遍历第一维
for i in range(full_array.shape[0]):
    row_segment = full_array[i, 0, :]
    
    # 检查当前段是否有 NaN
    if np.isnan(row_segment).any():
        print(f"警告: 第 {i} 行数据包含 NaN，np.max 可能会返回 NaN。")
    
    # 找到这个段中的最大值
    # 如果你想忽略NaN，可以使用 np.nanmax(row_segment)
    max_val = np.nanmax(row_segment)
    
    max_values.append(max_val)

# 将列表转换为 NumPy 数组
max_values = np.array(max_values)

print(f"\n找到的最大值数组形状: {max_values.shape}")
print("\n找到的20个最大值如下：")
print(f"{1.4-max_values:.3f}")