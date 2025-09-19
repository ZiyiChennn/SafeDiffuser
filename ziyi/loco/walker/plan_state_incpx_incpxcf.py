import torch
import os
import numpy as np
from os.path import join

# 提供的 10 个里程碑数字
milestone_numbers = np.array([1.254654, 1.250145, 1.2515, 1.247525, 1.25258, 1.247683, 1.254855, 1.254187, 1.252626, 1.252102])

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径
obs_file_path = os.path.join(script_dir, "plan_observation.pt")

try:
    # --- 步骤 1: 加载和处理数据 ---
    obs_tensor = torch.load(obs_file_path)
    full_array = obs_tensor.numpy()
    
    # 提取用于分段的参考数据：所有批次、第一个时间步、第 0 列数据
    milestone_search_data = full_array[:, 0, 0]
    
    print(f"观察数据形状: {full_array.shape}")
    print(f"里程碑搜索数据形状: {milestone_search_data.shape}")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {obs_file_path}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载或处理文件时发生错误: {e}")
    exit()

# --- 步骤 2: 使用近似匹配找到里程碑数字的索引 ---
milestone_indices = []
tolerance = 1e-6 

for milestone in milestone_numbers:
    indices = np.where(np.abs(milestone_search_data - milestone) < tolerance)[0]
    
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

for i in range(len(milestone_indices) - 1):
    start_index = milestone_indices[i]
    end_index = milestone_indices[i + 1]
    
    segment = full_array[start_index:end_index, :, :]
    
    if segment.size > 0:
        # 计算 segment[:, :, 0] + 0.1 * segment[:, :, 9]
        weighted_sum = segment[:, :, 0] + 0.1 * segment[:, :, 9]
        
        # 找到加权和的最大值
        max_val_in_segment = np.max(weighted_sum)
        
        # 找到最大值在分段内的相对索引
        max_val_relative_indices = np.where(weighted_sum == max_val_in_segment)
        
        # 找到最大值在整个 full_array 中的全局批次索引和时间步索引
        max_val_global_batch_index = start_index + max_val_relative_indices[0][0]
        max_val_time_index = max_val_relative_indices[1][0]
        
        # 存储最大值所在行的最大值所在时间步的 17 维数据
        results[i, :] = full_array[max_val_global_batch_index, max_val_time_index, :]
    else:
        results[i, :] = np.nan
        results[i, 0] = np.nan

# 处理最后一个区间
last_start_index = milestone_indices[-1]
last_segment = full_array[last_start_index:, :, :]

if last_segment.size > 0:
    # 计算 segment[:, :, 0] + 0.1 * segment[:, :, 9]
    weighted_sum = last_segment[:, :, 0] + 0.1 * last_segment[:, :, 9]
    
    max_val_in_segment = np.max(weighted_sum)
    max_val_relative_indices = np.where(weighted_sum == max_val_in_segment)
    
    max_val_global_batch_index = last_start_index + max_val_relative_indices[0][0]
    max_val_time_index = max_val_relative_indices[1][0]
    
    results[-1, :] = full_array[max_val_global_batch_index, max_val_time_index, :]
else:
    results[-1, :] = np.nan

# --- 添加：打印每个段的批次个数 ---
print("\n--- 每个段的批次个数 ---")
print("-----------------------")

for i in range(len(milestone_indices) - 1):
    
   
    start_num = milestone_numbers[i]
    end_num = milestone_numbers[i + 1]


last_start_index = milestone_indices[-1]
last_segment_length = len(milestone_search_data) - last_start_index
last_milestone = milestone_numbers[-1]
print(f"第 10 段 ({last_milestone:.6f} 到结尾): {last_segment_length} 个批次")

# --- 步骤 4: 使用新创建的数组进行最终计算和打印 ---
print("\n--- 分段计算结果 ---")
print("--------------------")

for i in range(len(milestone_numbers) - 1):
    start_num = milestone_numbers[i]
    end_num = milestone_numbers[i + 1]
    
    final_value = 1.4 - results[i, 0] - 0.1 * results[i, 9]
    print(f"第 {i+1} 段 ({start_num:.6f} 到 {end_num:.6f} 之前):")
    print(f"  最终计算结果: {final_value:.6f}")

last_milestone = milestone_numbers[-1]
final_value = 1.4 - results[-1, 0] - 0.1 * results[-1, 9]
print(f"\n第 10 段 ({last_milestone:.6f} 到结尾):")
print(f"  最终计算结果: {final_value:.6f}")