import torch
import os
import numpy as np
from os.path import join

# 提供的 10 个里程碑数字
milestone_numbers = np.array([1.254654, 1.250145, 1.2515, 1.247525, 1.25258, 1.247683, 1.254855, 1.254187, 1.252626, 1.252102])

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径
obs_file_path = os.path.join(script_dir, "env_observation.pt")
action_file_path = os.path.join(script_dir, "plan_action.pt")

try:
    # --- 步骤 1: 加载和处理数据 ---
    obs_tensor = torch.load(obs_file_path).squeeze(1)
    action_tensor = torch.load(action_file_path)
    action_data = action_tensor  # 保持原始形状 [6487, 600, 6]
    
    print(f"观察数据形状: {obs_tensor.shape}")
    print(f"动作数据形状: {action_data.shape}")
    
    obs_milestone_column = obs_tensor[:, 0].numpy()
    
except FileNotFoundError:
    print("错误: 找不到 observation.pt 或 action.pt 文件。请确保文件路径正确。")
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
    current_search_segment = obs_milestone_column[start_search_index:]
    
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

# --- 步骤 3: 分段并计算最大的绝对值及其索引 ---
results = []
for i in range(len(milestone_indices) - 1):
    start_index = milestone_indices[i]
    end_index = milestone_indices[i+1]
    
    segment = action_data[start_index:end_index]  # 形状: [segment_length, 600, 6]
    
    if segment.numel() > 0:
        # 计算绝对值并找到最大值
        abs_segment = torch.abs(segment)
        abs_max_val, abs_max_idx = torch.max(abs_segment.flatten(), dim=0)
        
        # 将一维索引转换回三维索引 (时间步, 600维索引, 6维索引)
        segment_length, dim_600, dim_6 = segment.shape
        time_idx = abs_max_idx // (dim_600 * dim_6)
        dim_600_idx = (abs_max_idx % (dim_600 * dim_6)) // dim_6
        dim_6_idx = abs_max_idx % dim_6
        
        # 获取原始值
        original_val = segment[time_idx, dim_600_idx, dim_6_idx]
        
        results.append({
            "abs_max": abs_max_val.item(),
            "original_val": original_val.item(),
            "time_idx": time_idx.item(),
            "dim_600_idx": dim_600_idx.item(),
            "dim_6_idx": dim_6_idx.item()
        })
    else:
        results.append({
            "abs_max": None,
            "original_val": None,
            "time_idx": None,
            "dim_600_idx": None,
            "dim_6_idx": None
        })

# 处理最后一段
last_start_index = milestone_indices[-1]
last_segment = action_data[last_start_index:]  # 形状: [last_segment_length, 600, 6]

if last_segment.numel() > 0:
    abs_last_segment = torch.abs(last_segment)
    abs_max_val, abs_max_idx = torch.max(abs_last_segment.flatten(), dim=0)
    
    # 将一维索引转换回三维索引
    segment_length, dim_600, dim_6 = last_segment.shape
    time_idx = abs_max_idx // (dim_600 * dim_6)
    dim_600_idx = (abs_max_idx % (dim_600 * dim_6)) // dim_6
    dim_6_idx = abs_max_idx % dim_6
    
    # 获取原始值
    original_val = last_segment[time_idx, dim_600_idx, dim_6_idx]
    
    results.append({
        "abs_max": abs_max_val.item(),
        "original_val": original_val.item(),
        "time_idx": time_idx.item(),
        "dim_600_idx": dim_600_idx.item(),
        "dim_6_idx": dim_6_idx.item()
    })
else:
    results.append({
        "abs_max": None,
        "original_val": None,
        "time_idx": None,
        "dim_600_idx": None,
        "dim_6_idx": None
    })

# --- 步骤 4: 打印结果 ---
print("\n--- 动作数据分段最大绝对值及其维度 ---")
print("-----------------------------------")

for i in range(len(results)):
    current_result = results[i]
    
    if i < len(milestone_numbers) - 1:
        start_num = milestone_numbers[i]
        end_num = milestone_numbers[i+1]
        print(f"{i+1} 条的 )...")
        if current_result["abs_max"] is not None:
            print(f"     最大绝对值: {1 - current_result['abs_max']:.3f}")
        else:
            print("     段内无数据")
    else:
        last_milestone = milestone_numbers[-1]
        print(f"从 {last_milestone:.6f} 到结尾区间的...")
        if current_result["abs_max"] is not None:
            print(f"     最大绝对值: {1 - current_result['abs_max']:.3f}")
        else:
            print("     段内无数据")