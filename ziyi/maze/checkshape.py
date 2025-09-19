import torch
import torch
import os
import numpy as np
from os.path import join
# 假设你的 .pt 文件名为 'my_tensor.pt'

script_dir = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径
file_path1 = os.path.join(script_dir, "safe1_all.pt")
file_path2 = os.path.join(script_dir, "safe2_all.pt")


try:
    np.set_printoptions(precision=3, suppress=True)
 
    tensor1 = torch.load(file_path1)
    safe1 = np.round(tensor1[:,0].cpu().numpy(), 3)
 
    # 检查加载的数据是否是张量
    if isinstance(tensor1, torch.Tensor):
     
        print(f"safe1:{safe1}")
  

    
    tensor2 = torch.load(file_path2)
    safe2 = np.round(tensor2[:,0].cpu().numpy(), 3)
 

   
    
    # 检查加载的数据是否是张量
    if isinstance(tensor2, torch.Tensor):
     
        print(f"safe2:{safe2}")

except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path1}'。请检查路径和文件名。")
except Exception as e:
    print(f"加载文件时发生错误: {e}")