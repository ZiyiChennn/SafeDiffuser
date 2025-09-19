import pdb
import os
from os.path import join
# os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin')
# os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515')
# python scripts/plan_guided.py --dataset walker2d-medium-expert-v2 --logbase logs


import diffuser.sampling as sampling
import diffuser.utils as utils

##for debug---------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_DIR, '..'))
 

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
##----------------------------------------------




#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


## for debug argument ----------------------------

#args.dataset = "walker2d-medium-replay-v2"
#args = Parser().read_config(args, 'plan')
#args.logbase = os.path.join(BASE_DIR,"logs/pretrained")
#args.loadbase = os.path.join(BASE_DIR,"logs/pretrained")

##args.horizon = "600"
#args.diffusion_loadpath = "diffusion/defaults_H32_T20"
#args.value_loadpath = "values/defaults_H32_T20_d0.997"
##print("args.diffusion_loadpath=",args.diffusion_loadpath)
##print("args.dataset=",args.dataset)
## -----------------------------------------------


args.batch_size = 1
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)


value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# import pdb; pdb.set_trace()
# data = {'norm_obs': dataset.fields.normed_observations, 'norm_act': dataset.fields.normed_actions, 'len': dataset.fields.path_lengths}
# import pickle
# output = open('./scripts/data_walker2d_medium.pkl', 'wb') 
# pickle.dump(data, output)
# output.close()


## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)  

#ln -s /usr/lib/nvidia /usr/lib/nvidia-515 # to address the issue: ERROR: Shadow framebuffer is not complete, error 0x8cd7 

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env
comp_time = []
safety = []
scores = []
import time
import torch
import random
import numpy as np

#-----------------------------ziyi---------------------------------------------#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

mode = 14 # Change this to 1-12 as needed
modes = {
    1: "diffuser",
    2: "GD",
    3: "Shield",
    4: "invariance",
    5: "invariance_cf",
    6: "invariance_cpx",
    7: "invariance_cpx_cf",


    8:  "diffuser_hopper", #0
    9:  "GD_hopper", #1
    10: "Shield_hopper",#2
    11: "invariance_hopper",#3
    12: "invariance_hopper_cf",#4
    13: "invariance_hopper_cpx",#5
    14: "invariance_hopper_cpx_cf",#6
   
}
sub_dir = join(args.savepath, modes.get(mode, "default"))
makedirs(sub_dir)

plan_observation = []
plan_action = []
env_observation = []
#--------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
#------------------------------------确定种子------------------------------------#
fixed_seeds = list(range(10))

#-------------------------------------------------------------------------------#
for kk in range(10):
# for kk in range(1): #测bmin
    seed = fixed_seeds[kk]
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation = env.reset()


    ## observations for rendering
    rollout = [observation.copy()]
    total_reward = 0
    for t in range(args.max_episode_length):
        
        # if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        
        conditions = {0: observation}

        # utils.colab.run_diffusion(diffusion, dataset, observation, n_samples=1, device=args.device, horizon=320, guide=guide, sample_fn=sampling.n_step_guided_p_sample
        #     )
        start = time.time()
        #samples.trajectory
        action, samples, diffusion, b_min = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        #------------------------------------ziyi---------------------------------
        e_observation = []
        e_observation.append(observation)
        env_observation.append(np.array(e_observation))


        p_observation = samples.observations[0]
        p_action = samples.actions[0]

        plan_observation.append(p_observation)
        plan_action.append(p_action)
        
        #------------------------------------ziyi----------------------------------
        end = time.time()
        if t == 0:
            safety.append(b_min.cpu().numpy())
            comp_time.append(end-start)
        ## execute action in environment
        
        # print(f"t: {t} | safety: {b_min.cpu().numpy():.4f}") #测bmin

        next_observation, reward, terminal, _ = env.step(action)

        ## print reward and score
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        #print(
        #   f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #   f'values: {samples.values} | scale: {args.scale}',
        #    flush=True,
        #)
        #--------------------------------------------------------
        #-------------------------固定随机种子看obervation和noise--------------------
        #---------------walker----------------------------------
        print(
            f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values} | scale: {args.scale}',
            flush=True,
        )
        #--------------------------------------------------------
        #---------------hopper----------------------------------
        '''print(
            f'step: {kk}/10 | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values} | scale: {args.scale} | oberservation[0]:{observation[0]:.3f}| oberservation[1]:{observation[1]:.3f}|oberservation[2]:{observation[2]:.3f}|oberservation[3]:{observation[3]:.3f}|oberservation[4]:{observation[4]:.3f}|oberservation[5]:{observation[5]:.3f}| oberservation[6]:{observation[6]:.3f}| oberservation[7]:{observation[7]:.3f}| oberservation[8]:{observation[8]:.3f}| oberservation[9]:{observation[9]:.3f}| oberservation[10]:{observation[10]:.3f}| action[0]:{action[0]:.3f}| action[1]:{action[1]:.3f}| action[2]:{action[2]:.3f}',
            flush=True,
        )'''
        #--------------------------------------------------------

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples, state, rollout, diffusion)

        if terminal:
            break

        observation = next_observation
    scores.append(score)

## write results to json file at `args.savepath`
# logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
import numpy as np
comp_time = np.array(comp_time)
safety = np.array(safety)
scores = np.array(scores)
 
#--------------------ziyi----------------------------------
makedirs(sub_dir)
torch.save(torch.tensor(np.array(env_observation)), join(sub_dir, 'env_observation.pt'))
torch.save(torch.tensor(np.array(plan_observation)), join(sub_dir, 'plan_observation.pt'))
torch.save(torch.tensor(np.array(plan_action)), join(sub_dir, 'plan_action.pt'))


import yaml

results = {
    
  
    "safety": float(np.min(safety)),
    "score_mean": float(np.mean(scores)),
    "score_std": float(np.std(scores)),
    "computation_time_mean": float(np.mean(comp_time)),
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
#--------------------ziyi-------------------------------------
print("safety: ", np.min(safety))
print("score mean: ", np.mean(scores))
print("score std: ", np.std(scores))
print("computation time: ", np.mean(comp_time))