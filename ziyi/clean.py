import torch
import re
import glob
import numpy as np
import os
import yaml



dataset_1= "walker"
dataset_2= "hopper"

level_1 = "expert"
level_2 = "replay"
level_3 = "medium"

'''diffuer_1="diffuser"
diffuer_2="GD"
diffuer_3="Shield"
diffuer_4="invariance"
diffuer_5="invariance_cf"
diffuer_6="invariance_cpx"
diffuer_7="invariance_cpx_cf"'''

diffuser_types = { # 将diffuser变量放入一个字典，方便迭代
    "diffuser_1": "diffuser",
    "diffuser_2": "GD",
    "diffuser_3": "Shield",
    "diffuser_4": "invariance",
    "diffuser_5": "invariance_cf",
    "diffuser_6": "invariance_cpx",
    "diffuser_7": "invariance_cpx_cf"
}

'''current_dataset = dataset_2
curren_level = level_2
current_diffuser = diffuer_7

log_dir = f"tabelle/{current_dataset}/{curren_level}"
log_files = glob.glob(os.path.join(log_dir, f"{current_dataset}__{curren_level}__{current_diffuser}_1000_fn.log"))

# 正则表达式
step_t_pattern = r"step:\s*(\d+)/\d+\s*\|\s*t:\s*(\d+)"
obs_pattern = r"oberservation\[(\d+)\]:(-?\d+\.\d+)"
act_pattern = r"action\[(\d+)\]:(-?\d+\.\d+)"



# 用于解析日志内容的正则表达式
patterns = {
    # 提取 env, horizon, max_path_length
    "env_horizon_max_path": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.SequenceDataset'>\s*?"
        r"\s*env:\s*([a-zA-Z0-9_-]+)\s*?"
        r"\s*horizon:\s*(\d+)\s*?"
        r"\s*max_path_length:\s*(\d+)",
        re.DOTALL
    ),
    # 提取 cond_dim (更名为 state_dim)
    "cond_dim": re.compile(
        r"Config: <class 'diffuser\.models\.temporal\.TemporalUnet'>.*?"
        r"\s*cond_dim:\s*(\d+)",
        re.DOTALL
    ),
    # 提取 action_dim
    "action_dim": re.compile(
        r"Config: <class 'diffuser\.models\.diffusion\.GaussianDiffusion'>.*?"
        r"\s*action_dim:\s*(\d+)",
        re.DOTALL
    ),
    # 提取 n_timesteps (更名为 diffusion_denoise_time_step)
    "n_timesteps": re.compile(
        r"Config: <class 'diffuser\.models\.diffusion\.GaussianDiffusion'>.*?"
        r"\s*n_timesteps:\s*(\d+)",
        re.DOTALL
    ),
    # 提取 diffusion_normalizer
    "diffusion_normalizer": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.SequenceDataset'>.*?"
        r"\s*normalizer:\s*([a-zA-Z0-9]+)",
        re.DOTALL
    ),
    # 提取 value_normalizer
    "value_normalizer": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.ValueDataset'>.*?"
        r"\s*normalizer:\s*([a-zA-Z0-9]+)",
        re.DOTALL
    ),
    # 提取 Trainer 配置块本身 (不提取具体参数，而是整个块)
    "trainer_block": re.compile(
        r"Config: <class 'diffuser\.utils\.training\.Trainer'>(.*?)(?=Config: <class 'diffuser\.|\[ datasets\/buffer \]|\[ models\/temporal \]|\[ utils\/config \]|Loaded config from|\Z)",
        re.DOTALL
    ),
    # 用于在 Trainer 块内提取具体参数
    "ema_decay_in_block": re.compile(r"ema_decay:\s*([0-9.]+)", re.DOTALL),
    "train_batch_size_in_block": re.compile(r"train_batch_size:\s*(\d+)", re.DOTALL),
    "train_lr_in_block": re.compile(r"train_lr:\s*([0-9.]+)", re.DOTALL),

    # 提取 value_termination_penalty (来自 ValueDataset)
    "value_termination_penalty": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.ValueDataset'>.*?"
        r"\s*termination_penalty:\s*([-0-9.]+)", # 可以是负数或浮点数
        re.DOTALL
    ),
    # 提取 max_n_epsiode (从 replay buffer 的日志行中提取)
    "max_n_epsiode": re.compile(
        r"Finalized replay buffer \| (\d+) episodes"
    ),

    #提取 safety, score mean, score std, computation time
    "safety": re.compile(r"safety:\s*([-0-9.]+)", re.IGNORECASE), # IGNORECASE for robustness
    "score_mean": re.compile(r"score mean:\s*([-0-9.]+)", re.IGNORECASE),
    "score_std": re.compile(r"score std:\s*([-0-9.]+)", re.IGNORECASE),
    "computation_time": re.compile(r"computation time:\s*([-0-9.]+)", re.IGNORECASE),

    # 用于提取 step 和 t 的正则表达式 (你的原有模式)
    "step_t": re.compile(r"step:\s*(\d+)/\d+\s*\|\s*t:\s*(\d+)"),
    "obs": re.compile(r"oberservation\[(\d+)\]:(-?\d+\.\d+)"),
    "act": re.compile(r"action\[(\d+)\]:(-?\d+\.\d+)")
}

for log_path in log_files:
    
    
    # 获取文件名，不含扩展名
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    
    # 初始化字典，用于存储此日志文件的配置
    extracted_config = {}

    # 读取整个日志文件内容，以便使用正则匹配多行
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # --- 从日志中提取参数 ---
        
        # 提取 env, horizon, max_path_length (并赋值给 max_t)
        match = patterns["env_horizon_max_path"].search(log_content)
        if match:
            extracted_config['env'] = match.group(1).strip()
            extracted_config['horizon'] = int(match.group(2))
            extracted_config['max_path_length'] = int(match.group(3))
            max_t = extracted_config['max_path_length'] # 用于数据处理
        else:
            
            print(f"警告: {log_path} 中缺少关键维度 max_path_length，数据处理可能失败。")
            max_t = 1000 # 仍然需要一个值来初始化数组，即使它不会被保存到 config

        # 提取 cond_dim (更名为 state_dim，并赋值给 obs_dim)
        match = patterns["cond_dim"].search(log_content)
        if match:
            extracted_config['state_dim'] = int(match.group(1))
            obs_dim = extracted_config['state_dim'] # 用于数据处理
        else:
            print(f"警告: {log_path} 中缺少关键维度 cond_dim，数据处理可能失败。")
            obs_dim = 11 # 仍然需要一个值来初始化数组

        # 提取 action_dim (并赋值给 action_dim_data)
        match = patterns["action_dim"].search(log_content)
        if match:
            extracted_config['action_dim'] = int(match.group(1))
            action_dim_data = extracted_config['action_dim'] # 用于数据处理
        else:
            print(f"警告: {log_path} 中缺少关键维度 action_dim，数据处理可能失败。")
            action_dim_data = 3 # 仍然需要一个值来初始化数组
            
        # 提取 n_timesteps (更名为 diffusion_denoise_time_step，并赋值给 num_steps)
        match = patterns["n_timesteps"].search(log_content)
        if match:
            extracted_config['diffusion_denoise_time_step'] = int(match.group(1))
            num_steps = extracted_config['diffusion_denoise_time_step'] # 用于数据处理 (NumPy 数组的第一个维度)
        else:
            print(f"警告: {log_path} 中缺少关键维度 n_timesteps，数据处理可能失败。")
            num_steps = 10 # 仍然需要一个值来初始化数组

        # 提取 diffusion_normalizer
        match = patterns["diffusion_normalizer"].search(log_content)
        if match:
            extracted_config['diffusion_normalizer'] = match.group(1).strip()

        # 提取 value_normalizer
        match = patterns["value_normalizer"].search(log_content)
        if match:
            extracted_config['value_normalizer'] = match.group(1).strip()

        # --- 提取 Trainer 配置 (更精确的匹配) ---
        # 查找所有 Trainer 配置块
        trainer_blocks = patterns["trainer_block"].findall(log_content)
        
        # 遍历所有找到的 Trainer 块，尝试提取参数。
        # 只要找到就记录，不设置默认值。
        for block_content in trainer_blocks:
            # 提取 ema_decay
            match_ema_decay = patterns["ema_decay_in_block"].search(block_content)
            if match_ema_decay:
                extracted_config['ema_decay'] = float(match_ema_decay.group(1))
            
            # 提取 train_batch_size
            match_batch_size = patterns["train_batch_size_in_block"].search(block_content)
            if match_batch_size:
                extracted_config['train_batch_size'] = int(match_batch_size.group(1))

            # 提取 train_lr
            match_train_lr = patterns["train_lr_in_block"].search(block_content)
            if match_train_lr:
                extracted_config['train_lr'] = float(match_train_lr.group(1))
        
        # 提取 value_termination_penalty
        match = patterns["value_termination_penalty"].search(log_content)
        if match:
            extracted_config['value_termination_penalty'] = float(match.group(1))

        # 提取 max_n_epsiode
        match = patterns["max_n_epsiode"].search(log_content)
        if match:
            extracted_config['max_n_episode'] = int(match.group(1))

        safety_matches = patterns["safety"].findall(log_content)
        if safety_matches:
            extracted_config['safety'] = float(safety_matches[-1]) # Get the last found safety value
        else:
            print(f"Warning: Could not find 'safety' in {log_path}.")

        score_mean_matches = patterns["score_mean"].findall(log_content)
        if score_mean_matches:
            extracted_config['score_mean'] = float(score_mean_matches[-1])
        else:
            print(f"Warning: Could not find 'score mean' in {log_path}.")
        
        score_std_matches = patterns["score_std"].findall(log_content)
        if score_std_matches:
            extracted_config['score_std'] = float(score_std_matches[-1])
        else:
            print(f"Warning: Could not find 'score std' in {log_path}.")

        computation_time_matches = patterns["computation_time"].findall(log_content)
        if computation_time_matches:
            extracted_config['computation_time'] = float(computation_time_matches[-1])
        else:
            print(f"Warning: Could not find 'computation time' in {log_path}.")

        # --- 保存 .yaml 配置文件 ---
        config_yaml_path = os.path.join(log_dir, f"{base_name}_config.yaml")
        with open(config_yaml_path, 'w', encoding='utf-8') as yaml_f:
            yaml.dump(extracted_config, yaml_f, default_flow_style=False, sort_keys=False)
        print(f"配置已保存到: {config_yaml_path}")

    except FileNotFoundError:
        print(f"错误：日志文件 '{log_path}' 未找到。")
        continue # 跳过当前文件，处理下一个
    except Exception as e:
        print(f"处理 '{log_path}' 时发生错误: {e}")
        continue # 跳过当前文件，处理下一


# 初始化 NaN 填充数据结构
    num_steps = 10
    observations = np.full((num_steps, obs_dim, max_t), np.nan, dtype=np.float32)
    actions = np.full((num_steps, action_dim_data, max_t), np.nan, dtype=np.float32)

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "oberservation" not in line or "action" not in line:
                continue

            step_t_match = re.search(step_t_pattern, line)
            if not step_t_match:
                continue

            step = int(step_t_match.group(1))
            t = int(step_t_match.group(2))
            if step >= num_steps or t >= max_t:
                continue  # 忽略越界项

            obs_matches = re.findall(obs_pattern, line)
            act_matches = re.findall(act_pattern, line)

            for idx, val in obs_matches:
                i = int(idx)
                if i < obs_dim:
                    observations[step, i, t] = float(val)

            for idx, val in act_matches:
                i = int(idx)
                if i < action_dim_data:
                    actions[step, i, t] = float(val)
            
        
        
        

    # 保存
    obs_save_path = os.path.join(log_dir, f"{base_name}_observation.pt")
    act_save_path = os.path.join(log_dir, f"{base_name}_action.pt")

    torch.save(observations, obs_save_path)
    torch.save(actions, act_save_path)

    print(f"Saved: {obs_save_path}, {act_save_path}")
    '''

patterns = {
    "env_horizon_max_path": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.SequenceDataset'>\s*?"
        r"\s*env:\s*([a-zA-Z0-9_-]+)\s*?"
        r"\s*horizon:\s*(\d+)\s*?"
        r"\s*max_path_length:\s*(\d+)",
        re.DOTALL
    ),
    "cond_dim": re.compile(
        r"Config: <class 'diffuser\.models\.temporal\.TemporalUnet'>.*?"
        r"\s*cond_dim:\s*(\d+)",
        re.DOTALL
    ),
    "action_dim": re.compile(
        r"Config: <class 'diffuser\.models\.diffusion\.GaussianDiffusion'>.*?"
        r"\s*action_dim:\s*(\d+)",
        re.DOTALL
    ),
    "n_timesteps": re.compile(
        r"Config: <class 'diffuser\.models\.diffusion\.GaussianDiffusion'>.*?"
        r"\s*n_timesteps:\s*(\d+)",
        re.DOTALL
    ),
    "diffusion_normalizer": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.SequenceDataset'>.*?"
        r"\s*normalizer:\s*([a-zA-Z0-9]+)",
        re.DOTALL
    ),
    "value_normalizer": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.ValueDataset'>.*?"
        r"\s*normalizer:\s*([a-zA-Z0-9]+)",
        re.DOTALL
    ),
    "trainer_block": re.compile(
        r"Config: <class 'diffuser\.utils\.training\.Trainer'>(.*?)(?=Config: <class 'diffuser\.|\[ datasets\/buffer \]|\[ models\/temporal \]|\[ utils\/config \]|Loaded config from|\Z)",
        re.DOTALL
    ),
    "ema_decay_in_block": re.compile(r"ema_decay:\s*([0-9.]+)", re.DOTALL),
    "train_batch_size_in_block": re.compile(r"train_batch_size:\s*(\d+)", re.DOTALL),
    "train_lr_in_block": re.compile(r"train_lr:\s*([0-9.]+)", re.DOTALL),
    "value_termination_penalty": re.compile(
        r"Config: <class 'diffuser\.datasets\.sequence\.ValueDataset'>.*?"
        r"\s*termination_penalty:\s*([-0-9.]+)",
        re.DOTALL
    ),
    "max_n_epsiode": re.compile(
        r"Finalized replay buffer \| (\d+) episodes"
    ),
    "safety": re.compile(r"safety:\s*([-0-9.]+)", re.IGNORECASE),
    "score_mean": re.compile(r"score mean:\s*([-0-9.]+)", re.IGNORECASE),
    "score_std": re.compile(r"score std:\s*([-0-9.]+)", re.IGNORECASE),
    "computation_time": re.compile(r"computation time:\s*([-0-9.]+)", re.IGNORECASE),
    "step_t": re.compile(r"step:\s*(\d+)/\d+\s*\|\s*t:\s*(\d+)"),
    "obs": re.compile(r"oberservation\[(\d+)\]:(-?\d+\.\d+)"),
    "act": re.compile(r"action\[(\d+)\]:(-?\d+\.\d+)")
}



current_dataset = dataset_2
current_level = level_1
# --- 开始循环遍历 diffuser 类型 ---
for diffuser_key, current_diffuser in diffuser_types.items():
    print(f"\n--- Processing diffuser type: {diffuser_key} ({current_diffuser}) ---")

    log_dir = f"tabelle/{current_dataset}/{current_level}"
    # 使用 current_diffuser 变量来构建文件名
    log_files = glob.glob(os.path.join(log_dir, f"{current_dataset}__{current_level}__{current_diffuser}_1000_fn.log"))

    if not log_files:
        print(f"No log files found for {current_dataset}__{current_level}__{current_diffuser}. Skipping.")
        continue # 如果没有找到日志文件，跳过当前diffuser类型

    for log_path in log_files:
        print(f"正在处理: {log_path}")
        
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        
        extracted_config = {}

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # --- 从日志中提取参数 ---
            
            match = patterns["env_horizon_max_path"].search(log_content)
            if match:
                extracted_config['env'] = match.group(1).strip()
                extracted_config['horizon'] = int(match.group(2))
                extracted_config['max_path_length'] = int(match.group(3))
                max_t = extracted_config['max_path_length']
            else:
                print(f"警告: {log_path} 中缺少关键维度 max_path_length，数据处理可能失败。")
             

            match = patterns["cond_dim"].search(log_content)
            if match:
                extracted_config['state_dim'] = int(match.group(1))
                obs_dim = extracted_config['state_dim']
            else:
                print(f"警告: {log_path} 中缺少关键维度 cond_dim，数据处理可能失败。")
          

            match = patterns["action_dim"].search(log_content)
            if match:
                extracted_config['action_dim'] = int(match.group(1))
                action_dim_data = extracted_config['action_dim']
            else:
                print(f"警告: {log_path} 中缺少关键维度 action_dim，数据处理可能失败。")
    
                
            match = patterns["n_timesteps"].search(log_content)
            if match:
                extracted_config['diffusion_denoise_time_step'] = int(match.group(1))
                num_steps = extracted_config['diffusion_denoise_time_step']
            else:
                print(f"警告: {log_path} 中缺少关键维度 n_timesteps，数据处理可能失败。")
            

            match = patterns["diffusion_normalizer"].search(log_content)
            if match:
                extracted_config['diffusion_normalizer'] = match.group(1).strip()

            match = patterns["value_normalizer"].search(log_content)
            if match:
                extracted_config['value_normalizer'] = match.group(1).strip()

            trainer_blocks = patterns["trainer_block"].findall(log_content)
            
            for block_content in trainer_blocks:
                match_ema_decay = patterns["ema_decay_in_block"].search(block_content)
                if match_ema_decay:
                    extracted_config['ema_decay'] = float(match_ema_decay.group(1))
                
                match_batch_size = patterns["train_batch_size_in_block"].search(block_content)
                if match_batch_size:
                    extracted_config['train_batch_size'] = int(match_batch_size.group(1))

                match_train_lr = patterns["train_lr_in_block"].search(block_content)
                if match_train_lr:
                    extracted_config['train_lr'] = float(match_train_lr.group(1))
            
            match = patterns["value_termination_penalty"].search(log_content)
            if match:
                extracted_config['value_termination_penalty'] = float(match.group(1))

            match = patterns["max_n_epsiode"].search(log_content)
            if match:
                extracted_config['max_n_episode'] = int(match.group(1))

            # 提取 safety, score mean, score std, computation time
            safety_matches = patterns["safety"].findall(log_content)
            if safety_matches:
                extracted_config['safety'] = float(safety_matches[-1])
            else:
                print(f"Warning: Could not find 'safety' in {log_path}.")

            score_mean_matches = patterns["score_mean"].findall(log_content)
            if score_mean_matches:
                extracted_config['score_mean'] = float(score_mean_matches[-1])
            else:
                print(f"Warning: Could not find 'score mean' in {log_path}.")
            
            score_std_matches = patterns["score_std"].findall(log_content)
            if score_std_matches:
                extracted_config['score_std'] = float(score_std_matches[-1])
            else:
                print(f"Warning: Could not find 'score std' in {log_path}.")

            computation_time_matches = patterns["computation_time"].findall(log_content)
            if computation_time_matches:
                extracted_config['computation_time'] = float(computation_time_matches[-1])
            else:
                print(f"Warning: Could not find 'computation time' in {log_path}.")

            
                
            # 保存 .yaml 配置文件
            config_yaml_path = os.path.join(log_dir, f"{base_name}_config.yaml")
            with open(config_yaml_path, 'w', encoding='utf-8') as yaml_f:
                yaml.dump(extracted_config, yaml_f, default_flow_style=False, sort_keys=False)
            print(f"配置已保存到: {config_yaml_path}")

        except FileNotFoundError:
            print(f"错误：日志文件 '{log_path}' 未找到。")
            continue
        except Exception as e:
            print(f"处理 '{log_path}' 时发生错误: {e}")
            continue

        # --- 你的现有数据提取和保存逻辑 ---
        observations = np.full((num_steps, obs_dim, max_t), np.nan, dtype=np.float32)
        actions = np.full((num_steps, action_dim_data, max_t), np.nan, dtype=np.float32)

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "oberservation" not in line and "action" not in line:
                    continue

                step_t_match = patterns["step_t"].search(line)
                if not step_t_match:
                    continue

                step = int(step_t_match.group(1))
                t = int(step_t_match.group(2))
                
                if step >= num_steps or t >= max_t:
                    continue

                obs_matches = patterns["obs"].findall(line)
                act_matches = patterns["act"].findall(line)

                for idx, val in obs_matches:
                    i = int(idx)
                    if i < obs_dim:
                        observations[step, i, t] = float(val)

                for idx, val in act_matches:
                    i = int(idx)
                    if i < action_dim_data:
                        actions[step, i, t] = float(val)
                
        # 保存 .pt 文件
        obs_save_path = os.path.join(log_dir, f"{base_name}_observation.pt")
        act_save_path = os.path.join(log_dir, f"{base_name}_action.pt")

        torch.save(observations, obs_save_path)
        torch.save(actions, act_save_path)

        print(f"已保存: {obs_save_path}, {act_save_path}")

print("\n--- All diffuser types processed. ---")
