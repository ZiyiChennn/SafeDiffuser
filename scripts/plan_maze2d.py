import json
import numpy as np
from os.path import join
import pdb
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch
import random
# import matplotlib
# matplotlib.use("Agg")  # 强制使用非GUI后端


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy


#--------------------------------------ziyi---------------------------------------
mode = 1# Change this to 1-12 as needed
modes = {
    1: "diffuser",
    2: "GD",
    3: "Shield",
    4: "invariance",#local trap
    5: "invariance_cf",#local trap
    6: "invariance_relax",
    7: "invariance_relax_cf",
    8: "invariance_time",
    9: "invariance_time_cf",
    10: "invariance_relax_narrow",
    11: "GD_umaze",
    12: "Shield_umaze",
    13: "invariance_umaze",
    14: "invariance_umaze_relax",
    15: "diffuser_umaze"
}
sub_dir = join(args.savepath, modes.get(mode, "default"))
makedirs(sub_dir)
#-------------------------------------------------------------------------------
#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
score_batch = []
comp_time = []
elbo_batch = []
success = 0

#--------------ziyi---customize-------------------------------
all_observations = []
all_targets = []
all_diffusion_sm = []
all_actions = []
all_observations_env = []
all_actions_env = []

#all_next_observations = []
#--------------ziyi---customize-------------------------------

####--------------------ziyi---------fix---------------observation------------initial position--------------------------------------------------------------
    

observation_large = {
    0 : [ 1.0195254,   2.37341661,  0,          0        ],
    1 : [1.04110535,   1.998743,   0,         0        ],
    2 : [4.06317071e+00,  9.90054770e+00, -2.32503077e-01, -2.18791664e-02],
    3 : [7.04593109,  7.93513112, -0.0544259,  -0.03163002],
    4 : [0.96946192,  2.22096705,  0,          0        ],
    5 : [2.02943790e+00,  1.00230770e+01,  9.40122978e-03, -7.43499249e-02],
    6 : [6.0371084,   8.03009186, -0.10096182, -0.02091756],
    7 : [6.9270193,   9.04429767,  0.02146591,  0.03553727],
    8 : [3.07789757,  6.0868087,   0.14934311, -0.12590655],
    9 : [2.96437388, 10.01886001,  0.07813114,  0.02644556]
    }
observation_umaze = {
    0 : [ 2.95395734,  2.9081947,   0.01049001, -0.05356694],
    1 : [ 3.08255112,  2.02132716,  0.0947081,  -0.07037352], 
    2 : [ 3.06317071,  0.9005477,  -0.23250308, -0.02187917],
    3 : [ 1.04593109,  0.93513112, -0.0544259,  -0.03163002],
    4 : [ 0.98453744,  1.90566393,  0.13664635, -0.06651947],
    5 : [ 2.0294379,   3.02307702,  0.00940123, -0.07434992],
    6 : [ 3.0371084,   3.03009186, -0.10096182, -0.02091756],
    7 : [ 0.9270193,   1.04429767,  0.02146591,  0.03553727],
    8 : [ 2.07789757,  3.0868087,   0.14934311, -0.12590655],
    9 : [ 2.96437388,  1.01886001,  0.07813114,  0.02644556]
    }
####-----------------------------ziyi---------fix---------------observation------------initial position----------------------------------------------------------

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#seeds = [7]
import time
for iter in range(10):   # num of testing runs
    print("step: ", iter, "/10")


    seed = seeds[iter]
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    # observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    
    dataset_choice = 0
    if dataset_choice == 0:
        selected_dict = observation_large
    elif dataset_choice == 1:
        selected_dict = observation_umaze
 

    # 从选定的字典中获取 observation
    initial_observation_list = selected_dict[iter]

    # 将列表转换为 NumPy 数组
    initial_observation = np.array(initial_observation_list)
    # ---------------ziyi debug-----large GD第7组
    #initial_observation = np.array([ 0.9270193,   1.04429767,  0.02146591,  0.03553727])
    #######-------------------------
    observation = initial_observation
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]


    #--------------------------ziyi---------------------------
    actions_iter = []  # Collect actions for this iteration
    next_observations_iter = []  # New: Collect next_observations for this iteration
    #-------------------------ziyi-------------------------


    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            start = time.time()
            action, samples, diffusion_paths, safe1, safe2, elbo = policy(cond, batch_size=args.batch_size)
            end = time.time()
            comp_time.append(end-start)
            elbo_batch.append(elbo)
            
    #############################       single test
            # cond[0] = observation
            # action, samples, diffusion_paths, safe1, safe2 = policy(cond, batch_size=args.batch_size)  #policy.normalizer.normalizers['observations'].mins
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_sm = diffusion_paths[0]

            #------------------------ziyi-------------------------------
            
            all_observations.append(sequence)
            all_diffusion_sm.append(diffusion_sm)

            all_actions.append(actions)
            #------------------------ziyi-------------------------------

            
            ##################################################save videos/images
            fullpath = join(sub_dir, f'{iter}.png')
            renderer.composite(fullpath, samples.observations, ncol=1)
            #########################################s################# 8/3/2023
            # diffusion_sm = smooth(diffusion_paths)    # smooth the generated traj.
            diffusion_sm = diffusion_paths            # do not smooth the generated traj.

            #-----------------customize-------------------------------
            #renderer.render_diffusion(join(sub_dir, f'diffusion.mp4'), diffusion_sm)
            #---------------------------------------------------------

            # makedirs(join(args.savepath, 'trap'))
            # fullpath = join(args.savepath, f'trap/{iter}.png')
            # renderer.composite(fullpath, samples.observations, ncol=1)

            diff_step = diffusion_sm.shape[0]  
            makedirs(join(sub_dir, 'png'))
            for kk in range(diff_step):
                imgpath = join(sub_dir, f'png/{kk}.png')
                renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)
            ##################################################end saving videos/images

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        #------------------------------ziyi--------Collect action------------
        actions_iter.append(action)
        #------------------------------ziyi---------------------

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        #------------------------------ziyi--------Collect next_observation------------
        next_observations_iter.append(next_observation)
        #------------------------------ziyi---------------------

        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ###############################################################################################
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )
#--------------------ziyi target---
        if 'maze2d' in args.dataset:
             xy = next_observation[:2]
             goal = env.unwrapped._target
             print(
                 f'maze | pos: {xy} | goal: {goal}'
             )
#--------------
        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        ###############################################################################################
        # if t % args.vis_freq == 0 or terminal:
        #     fullpath = join(args.savepath, f'{t}.png')

        #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        #     # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        #     ## save rollout thus far
        #     renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)   ## debug

        #     # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        #     # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    if reward > 0.95:
        success = success + 1

    # score = 0

    #---------------------ziyi------------Collect data for this iteration---------------------------
    
    all_targets.append(target)
    all_actions_env.append(np.array(actions_iter))
    all_observations_env.append(np.array(next_observations_iter))  # New: Append collected next_observations
    #----------------------ziyi---------------------------------------------------------------------

    safe1_batch.append(torch.cat([safe1[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe1.device)], dim = 1))
    safe2_batch.append(torch.cat([safe2[-1].unsqueeze(0).unsqueeze(0), torch.tensor(score).unsqueeze(0).unsqueeze(0).to(safe2.device)], dim = 1))
    score_batch.append(score)
    # logger.finish(t, env.max_episode_steps, score=score, value=0)


#-----------------------------ziyi----------------------------------------------- 
# Convert collected data to tensors and save (use sub_dir)
makedirs(sub_dir)  # Ensure sub_dir exists

# Save observations
torch.save(torch.tensor(np.array(all_observations)), join(sub_dir, 'observations.pt'))

# Save observations_env
torch.save(torch.tensor(np.array(all_observations_env)), join(sub_dir, 'observations_env.pt'))

# Save targets
#torch.save(torch.tensor(np.array(all_targets)), join(sub_dir, 'targets.pt'))

# Save diffusion_sm
#torch.save(torch.tensor(np.array(all_diffusion_sm)), join(sub_dir, 'diffusion_sm_all.pt'))

# Save safe1_batch
torch.save(torch.cat(safe1_batch, dim=0), join(sub_dir, 'safe1_all.pt'))

# Save safe2_batch
torch.save(torch.cat(safe2_batch, dim=0), join(sub_dir, 'safe2_all.pt'))

# Save scores
torch.save(torch.tensor(score_batch), join(sub_dir, 'scores.pt'))

# Save actions
torch.save(torch.tensor(np.array(all_actions)), join(sub_dir, 'actions.pt'))

# Save actions_adjusted
torch.save(torch.tensor(np.array(all_actions_env)), join(sub_dir, 'actions_env.pt'))

# New: Save next_observations
#torch.save(torch.tensor(np.array(all_next_observations)), join(sub_dir, 'next_observations.pt'))
#--------------------ziyi--------------------------------------------------------------------

elbo_batch = np.array(elbo_batch)
print("elbo mean: ", np.mean(elbo_batch))
print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
safe1_batch = torch.cat(safe1_batch, dim=0)
safe2_batch = torch.cat(safe2_batch, dim=0)
comp_time = np.array(comp_time)

#---------------Ziyi---------------------------
# ... (你的所有代码，包括打印 elbo, score, etc. 的部分)

# -------------------- ziyi: 保存统计数据到 YAML --------------------
import yaml

# 组织需要保存的数据
results = {
    "elbo_mean": float(np.mean(elbo_batch)),
    "elbo_std": float(np.std(elbo_batch)),
    "safe1_min": float(torch.min(safe1_batch[:, 0]).cpu().numpy()),
    "safe2_min": float(torch.min(safe2_batch[:, 0]).cpu().numpy()),
    "score_mean": float(np.mean(score_batch)),
    "score_std": float(np.std(score_batch)),
    "computation_time_mean": float(np.mean(comp_time)),
    "success_rate": float(success)
}

# 组织需要保存的参数
# 确保这些参数在你的 args 对象中存在


# 合并所有数据
data_to_save = {
    "results": results,

}

# 构建 YAML 文件的完整路径
yaml_path = join(sub_dir, 'summary.yaml')

# 将数据保存为 YAML 文件
try:
    with open(yaml_path, 'w') as f:
        yaml.dump(data_to_save, f, sort_keys=False) # sort_keys=False 保持字典的原始顺序
    print(f"Summary saved to {yaml_path}")
except Exception as e:
    print(f"Error saving YAML file: {e}")

# ... (代码的其他部分，如 exit() )
#----------------Ziyi----------------------------
print("safe1: ", torch.min(safe1_batch[:,0]).cpu().numpy())
print("safe2: ", torch.min(safe2_batch[:,0]).cpu().numpy())
print("score mean: ", np.mean(score_batch))
print("score std: ", np.std(score_batch))
print("computation time: ", np.mean(comp_time))
print("success rate: ", success)

  #-----------------customize-------------------------------
 
  #-----------------customize-------------------------------
exit()

#-----------------customize-------------------------------
'''import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 4), facecolor='white')
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
plt.show(block=False)

ax1.cla()
ax1.set_title('Trajectories')
ax1.set_xlabel('score')
ax1.set_ylabel('min. S-spec')
ax1.plot(safe1_batch.cpu().numpy()[:,1], safe1_batch.cpu().numpy()[:,0], 'r*', label = 'ground truth')

ax2.cla()
ax2.set_title('Trajectories')
ax2.set_xlabel('score')
ax2.set_ylabel('min. C-spec')
ax2.plot(safe2_batch.cpu().numpy()[:,1], safe2_batch.cpu().numpy()[:,0], 'r*', label = 'ground truth')

imgpath = join(args.savepath, f'stat.png')

plt.savefig(imgpath)

import pdb; pdb.set_trace()

exit()'''
#-----------------customize-------------------------------





## save result as a json file
json_path = join(sub_dir, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

