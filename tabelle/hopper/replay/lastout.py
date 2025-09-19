import torch
import numpy as np
import os

# 定义文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
pt_file_path = os.path.join(script_dir, "hopper__replay__invariance_cf_1000_fn_observation.pt") # 请将文件名替换为你的实际文件名
txt_file_path = os.path.join(script_dir, "output_data.txt")
max_values_file_path = os.path.join(script_dir, "max_values.txt")

try:
    # 步骤 1: 加载 .pt 文件并进行类型检查和转换
    print(f"正在加载文件: {pt_file_path}")
    loaded_data = torch.load(pt_file_path)

    if isinstance(loaded_data, torch.Tensor):
        array_data = loaded_data.numpy()
        print("数据已成功从 PyTorch 张量转换为 NumPy 数组。")
    elif isinstance(loaded_data, np.ndarray):
        array_data = loaded_data
        print("加载的数据已经是 NumPy 数组。")
    else:
        print(f"错误：不支持的数据类型 {type(loaded_data)}。")
        exit()

    # 确保数组有三维，以进行切片操作
    if array_data.ndim < 3:
        print("错误：数组维度小于 3，无法执行切片操作。")
        exit()

    # 步骤 2: 切片并交换维度
    # 只取第一维的前10个元素，第二维取0，第三维取全部
    sliced_data = array_data[:10, 0, :]
    print(f"切片后数组形状: {sliced_data.shape}")

    # 将第一维和第三维进行交换，得到 (1000, 10) 的数组
    transposed_data = sliced_data.T
    print(f"交换维度后数组形状: {transposed_data.shape}")
    
    # 将切片后的数据保存到 .txt 文件，并限制为三位小数
    np.savetxt(txt_file_path, transposed_data, fmt='%.3f')
    print(f"转换后的数据已成功保存到文件: {txt_file_path}")

    # --- 步骤 3: 找到每列的最大值并保存 ---
    print("\n正在计算每列的最大值...")
    # 使用 axis=0 沿列（1000行）找到最大值
    max_values_per_column = 1.5-np.nanmax(transposed_data, axis=0)
    
    for i, val in enumerate(max_values_per_column):
        print(f"第 {i+1} 个值: {val:.3f}")
    
    # 将这10个最大值保存到新文件，并限制为三位小数
    

except FileNotFoundError:
    print(f"错误：找不到文件 '{pt_file_path}'。请检查文件路径是否正确。")
except Exception as e:
    print(f"处理文件时发生错误: {e}")