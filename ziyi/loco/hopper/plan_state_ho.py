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
start_search_index = 0
for i, milestone in enumerate(milestone_numbers):
    # 从上一个找到的索引之后开始搜索
    current_search_segment = milestone_search_data[start_search_index:]
    
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


# --- 步骤 3: 分段计算并将结果存入 [10, 17] 数组 ---
results = np.zeros((10, 11))

for i in range(len(milestone_indices) - 1):
    start_index = milestone_indices[i]
    end_index = milestone_indices[i + 1]
    
    segment = full_array[start_index:end_index, :, :]
    
    if segment.size > 0:
       
        # 找到最大值
        max_val_in_segment = np.max(segment[:, :, 0])
        
        # 找到最大值在分段内的相对索引
        max_val_relative_indices = np.where(segment[:, :, 0] == max_val_in_segment)
        
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
    
    max_val_in_segment = np.max(last_segment[:, :, 0] )
    max_val_relative_indices = np.where(last_segment[:, :, 0]  == max_val_in_segment)
    
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
    
    final_value = 1.6 - results[i, 0]
    print(f"第 {i+1} 段 ):")
    print(f"  最终计算结果: {final_value:.3f}")

last_milestone = milestone_numbers[-1]
final_value = 1.6 - results[-1, 0]  # expert 1.6 replay 1.55 medium 1.6
print(f"\n第 10 段 ({last_milestone:.6f} 到结尾):")
print(f"  最终计算结果: {final_value:.3f}")