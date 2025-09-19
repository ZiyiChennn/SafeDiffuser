import torch
import os
import numpy as np
from os.path import join

# 提供的 10 个里程碑数字
milestone_numbers = np.array([1.254654, 1.250145, 1.2515, 1.247525, 1.25258, 1.247683, 1.254855, 1.254187, 1.252626, 1.252102])

script_dir = os.path.dirname(os.path.abspath(__file__))

obs_file_path = os.path.join(script_dir, "env_observation.pt")

try:
    
    # 加载 PyTorch 张量并移除大小为 1 的维度
    obs_tensor = torch.load(obs_file_path).squeeze(1)
    
    # 将整个张量转换为 NumPy 数组，以便于处理和存储
    full_array = obs_tensor.numpy()
    
    # 提取第一列 (索引 0) 用于分段
    arr1 = full_array[:, 0]
    
    print(f"观察数据形状: {full_array.shape}")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {obs_file_path}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载或处理文件时发生错误: {e}")
    exit()


milestone_indices = []
# 设置一个小的容差值，用于浮点数比较
tolerance = 1e-6 

for milestone in milestone_numbers:
    # 使用近似匹配来查找索引
    indices = np.where(np.abs(arr1 - milestone) < tolerance)[0]
    
    if indices.size > 0:
        milestone_indices.append(indices[0])
    else:
        print(f"警告: 在数组第一列中找不到近似匹配的数字 {milestone}")
        milestone_indices.append(None)

if None in milestone_indices:
    print("错误: 某些里程碑数字不存在，无法进行分段计算。")
    exit()

# --- 步骤 3: 分段计算并将结果存入 [10, 17] 数组 ---

results = np.zeros((10, 17))

# 遍历前 9 个区间
for i in range(len(milestone_indices) - 1):
    start_index = milestone_indices[i]
    end_index = milestone_indices[i+1]
    
    print(f"第 {i+1} 段 ({start_index} 到 {end_index}): 个批次")
    
    segment = arr1[start_index:end_index]
    if len(segment) > 0:
        max_val = np.max(segment)
        max_val_relative_index = np.where(segment == max_val)[0][0]
        max_val_batch_index = start_index + max_val_relative_index
        results[i, :] = full_array[max_val_batch_index, :]
   
    else:
        results[i, :] = np.nan
        results[i, 0] = np.nan

# 计算最后一个区间
last_start_index = milestone_indices[-1]



# 将最后一个区间的最大值存储到 results 数组的第一列
last_segment = arr1[last_start_index:]
if len(last_segment) > 0:
    max_val = np.nanmax(last_segment)
    max_val_relative_index = np.where(last_segment == max_val)[0][0]
    max_val_batch_index = last_start_index + max_val_relative_index
    results[-1, :] = full_array[max_val_batch_index, :]
    
else:
    results[-1, :] = np.nan
    results[-1, 0] = np.nan

# --- 步骤 4: 使用新创建的数组进行最终计算和打印 ---
print("\n--- 分段计算结果 ---")
print("--------------------")

for i in range(len(milestone_numbers) - 1):
    start_num = milestone_numbers[i]
    end_num = milestone_numbers[i+1]
    
    # 使用数组切片进行最终计算
    
    final_value = 1.4 - results[i, 0] # in in_cf shield
    print(f"第 {i+1}  段:")
    
    print(f"  最终计算结果: {final_value:.3f}")

# 打印最后一个区间的结果
last_milestone = milestone_numbers[-1]
final_value = 1.4 - results[-1, 0] 

print(f"\n第 10 段  到结尾):")

print(f"  最终计算结果: {final_value:.3f}")