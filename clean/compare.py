import torch
import os
import numpy as np
from os.path import join
# 假设你的文件是 'my_tensor.pt'



file_a = 'observations_diff.txt'
file_b = 'observations_shield.txt'


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(script_dir, file_a)
file_path2 = os.path.join(script_dir, file_b)


def compare(path1, path2):
    """一次性读取并比较两个文本文件的所有内容。"""
    try:
        with open(path1, 'r', encoding='utf-8') as file1, \
             open(path2, 'r', encoding='utf-8') as file2:

            # 逐行读取并比较
            for line_num, (line1, line2) in enumerate(zip(file1, file2), 1):
                if line1 != line2:
                    print("-" * 30)
                    print(f"文件在第 {line_num} 行内容不同。")
                    print(f"文件1：{line1.strip()}")
                    print(f"文件2：{line2.strip()}")
                    print("-" * 30)
                    return False
                else:
                    print("true")
                    return True
            
            # 检查文件行数是否相同
            if len(file1.readlines()) != 0 or len(file2.readlines()) != 0:
                print("两个文件行数不同。")
                return False
            
    except FileNotFoundError:
        print(f"错误：文件未找到。请检查路径：'{path1}' 或 '{path2}'")
        return False
    except Exception as e:
        print(f"比较文件时发生错误：{e}")
        return False
    

compare(file_path1, file_path2)