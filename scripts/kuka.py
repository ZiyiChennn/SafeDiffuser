
import torch
import pdb

import gym
import d4rl

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
# from denoising_diffusion_pytorch.temporal import TemporalMixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
#import environments
import sys
sys.path.append('/data/vision/billf/scratch/yilundu/pddlstream')


B_IsTrainFromPreTrained = True

#### dataset
H = 128
env = gym.make('hopper-medium-v2')
dataset = KukaDataset(H)
#----------------ziyidebug----------
timesteps = 100,   # number of steps 1000
#------------------ziyidebug---------

#renderer = KukaRenderer()

## dimensions
obs_dim = dataset.obs_dim

#### model
# model = Unet(
#     width = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

diffusion_path = f'logs/multiple_cube_kuka_conv_new_real2_{H}'
diffusion_epoch = 0

# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

# model = MixerUnet(
#     horizon = H,
#     transition_dim = obs_dim,
#     cond_dim = H,
#     dim = 32,
#     dim_mults = (1, 2, 4, 8),
# ).cuda()

model = TemporalUnet(
    horizon = H,
    transition_dim = obs_dim,
    cond_dim = H,
    dim = 128,
    dim_mults = (1, 2, 4, 8),
).cuda()


diffusion = GaussianDiffusion(
    model,
    channels = 1,
    image_size = (H, obs_dim),
    timesteps = 100,   # number of steps 1000
    loss_type = 'l1'    # L1 or L2
).cuda()

train_num_steps = 700000
resultfolder = f'/root/kuka_dataset/multiple_cube_kuka_convnew_real2'

trainer = Trainer(
    diffusion,
    dataset,
    None,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = train_num_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = f'{resultfolder}_{H}_{timesteps}',
)
if B_IsTrainFromPreTrained:
    trainer.load(200)
    trainer.set_result_folder(f'../kuka_dataset/pretrained_{H}_{timesteps}')


#### test
print('testing forward')
x = dataset[0][0].view(1, H, obs_dim).cuda()
mask = torch.zeros(1, H).cuda()

loss = diffusion(x, mask)
loss.backward()
print('done')
# pdb.set_trace()
####



trainer.train()
