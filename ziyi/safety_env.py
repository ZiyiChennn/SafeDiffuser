import numpy as np
import os
from os.path import join

milestone_numbers = np.array([1.254654, 1.250145, 1.2515, 1.247525, 1.25258, 1.247683, 1.254855, 1.254187, 1.252626, 1.252102])

script_dir = os.path.dirname(os.path.abspath(__file__))

names = {
    1: "diffuser_env_ob.txt",
    2: "gd_env_ob.txt",
    3: "invariance_env_ob.txt",
    4: "invariance_cf_env_ob.txt",
    5: "invariance_cpx_env_ob.txt",
    6: "invariance_cpx_cf_env_ob.txt",
    7: "shield_env_ob.txt"
}

try:
    file_path1 = os.path.join(script_dir, names[7])
    
    # Load the full 2D array [6418, 17]
    full_array = np.loadtxt(file_path1)
    
    # Extract the first column (for calculations)
    arr1 = full_array[:, 0]
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path1}。请检查路径。")
    exit()

# --- 步骤 1: 找到每个里程碑数字在第一维中的位置 ---
milestone_indices = []
for milestone in milestone_numbers:
    indices = np.where(arr1 == milestone)[0]
    if indices.size > 0:
        milestone_indices.append(indices[0])
    else:
        print(f"警告: 在数组的第一维中找不到数字 {milestone}")
        milestone_indices.append(None)

if None in milestone_indices:
    print("错误: 某些里程碑数字在大数组的第一维中不存在，无法进行计算。")
else:
   
    results = np.zeros((10, 17))
    
    # 遍历前 9 个区间
    for i in range(len(milestone_indices) - 1):
        start_index = milestone_indices[i]
        end_index = milestone_indices[i+1]
        
        # 将整个行数据（17列）存入 results 数组
        results[i, :] = full_array[start_index, :]
        
        # 将区间的最大值存储到 results 数组的第一列
        segment = arr1[start_index:end_index]
        if len(segment) > 0:
            results[i, 0] = np.max(segment)
        else:
            results[i, 0] = np.nan # 用 NaN 表示空区间
    
    # 计算最后一个区间
    last_start_index = milestone_indices[-1]
    
    # 将最后一行数据存入 results 数组
    results[-1, :] = full_array[last_start_index, :]
    
    # 将最后一个区间的最大值存储到 results 数组的第一列
    last_segment = arr1[last_start_index:]
    if len(last_segment) > 0:
        results[-1, 0] = np.max(last_segment)
    else:
        results[-1, 0] = np.nan
    
   
    print("\n计算结果:")
    print("--------------------")
    
    # 打印前 9 个区间的结果
    for i in range(len(milestone_numbers) - 1):
        start_num = milestone_numbers[i]
        end_num = milestone_numbers[i+1]
        
        
        #final_value = 1.35- results[i, 0] - 0.1 * results[i, 9]    # diffuser 1   gd 2   in_cpx 5   in_cpx_cf 6 
        final_value = 1.35 - results[i, 0]   #in 3   in_cf 4   shield 7
        
        print(f"从 {start_num} 到 {end_num} 之前区间的最大值: {results[i, 0]:.6f}, h_min: {final_value:.6f}")
    
    # 打印最后一个区间的结果
    last_milestone = milestone_numbers[-1]
    
    # 使用数组切片进行最终计算
    #final_value = 1.35 - results[-1, 0] - 0.1 * results[-1, 9]          # diffuser 1   gd 2   in_cpx 5    in_cpx_cf 6 
    final_value = 1.35 - results[-1, 0]     #in 3   in_cf 4   shield 7
    
    print(f"从 {last_milestone} 到数组结尾区间的最大值: {results[-1, 0]:.6f}, h_min: {final_value:.6f}")