import torch
import os
import numpy as np
from os.path import join

# --- File Loading and Squeezing ---
script_dir = os.path.dirname(os.path.abspath(__file__))

modes = {
    1: "env_observation.pt",
    2: "plan_action.pt",
    3: "plan_observation.pt",
}

# The file to load
file_path_to_load = os.path.join(script_dir, modes[1])

try:
    # Load the PyTorch tensor. The shape is [6418, 1, 17].
    env_observation = torch.load(file_path_to_load)
    
    # Remove the dimension of size 1 at index 1.
    # The new shape will be [6418, 17].
    squeezed_tensor = env_observation.squeeze(1)
    
    # Check the new shape to confirm
    print(f"Original shape: {env_observation.shape}")
#     print(f"Squeezed shape: {squeezed_tensor.shape}")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path_to_load}")
    exit()

# # --- Saving to a Text File ---
# # Convert the squeezed tensor to a NumPy array.
numpy_array = squeezed_tensor.numpy()

# # Specify the output filename and path.
names = {
    1: "env_observation.txt",
    2: "gd_env_ob.txt",
    3: "invariance_env_ob.txt",
    4: "invariance_cf_env_ob.txt",
    5: "invariance_cpx_env_ob.txt",
    6: "invariance_cpx_cf_env_ob.txt",
    7: "shield_env_ob.txt"
}

name = names[1]

output_filename = name
output_path = os.path.join(script_dir, output_filename)

# Save the NumPy array to a text file in the same directory as the script.
# `fmt='%.6f'` formats the numbers with 6 decimal places for clarity.
np.savetxt(output_path, numpy_array, fmt='%.6f', delimiter=' ', newline='\n')

print(f"\n数据已成功保存到: {output_path}")