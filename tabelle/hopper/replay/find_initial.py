import torch
import os

# 假设你的文件名为: file1.pt, file2.pt, ..., file5.pt
# 请根据你的实际文件名修改下面的列表
script_dir = os.path.dirname(os.path.abspath(__file__))
path1 = os.path.join(script_dir, "hopper__replay__GD_1000_fn_observation.pt")
path2 = os.path.join(script_dir, "hopper__replay__diffuser_1000_fn_observation.pt")

file_names = [
    path1,
    path2

    #"file3.pt",
    #"file4.pt",
    #"file5.pt",
]

for file_path in file_names:
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在。请检查文件名和路径。")
        exit()

# 加载并提取每个数组的第0维值
tensors = []
for file_path in file_names:
    try:
        tensor = torch.load(file_path).squeeze(1)
        tensors.append(tensor)
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
        exit()

# 将所有数组的第0维值转换为集合，用于快速查找共同值
sets_of_values = [set(t[:, 0].tolist()) for t in tensors]

# 找出所有集合的交集
if not sets_of_values:
    print("没有找到任何文件，无法进行比较。")
    exit()

# 使用集合的 intersection 方法找出所有集合的共同元素
common_values = sets_of_values[0]
for s in sets_of_values[1:]:
    common_values = common_values.intersection(s)

# --- 核心修改部分 ---
# 根据第一个数组中共同值的出现顺序进行排列
ordered_common_values = []
# 遍历第一个数组的第0维
if tensors:
    for value in tensors[0][:, 0].tolist():
        # 如果这个值在共同值集合中，并且还没有被添加到有序列表中
        # 这个额外的检查可以防止重复值被多次添加
        if value in common_values and value not in ordered_common_values:
            ordered_common_values.append(value)

# 打印结果
if ordered_common_values:
    print("--- 在所有数组的第0维中找到的共同值（按第一个数组的顺序排列） ---")
    for value in ordered_common_values:
        print(f"{value:.6f}")
else:
    print("在所有数组的第0维中，没有找到任何共同值。")