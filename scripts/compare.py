import os
import numpy as np
import diffuser.sampling as sampling
import diffuser.utils as utils



#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

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

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env

# Initialize environment
observation = env.reset()

# Generate one trajectory
conditions = {0: observation}
action, samples, diffusion, b_min = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

# Extract actions and observations from trajectories

actions = samples.actions  # (1, 600, 6)
observations = samples.observations  # (1, 600, 17)

# Loop over horizon (600 steps)
for h in range(args.horizon):
    # Use action from trajectories[h]
    action_h = actions[0, h]  # (6,)

    # Execute action in environment
    next_observation, reward, terminal, _ = env.step(action_h)

    # Compare next_observation with trajectories[h+1].observations
    if h < args.horizon - 1:  # Ensure h+1 is within horizon
        predicted_observation = observations[0, h+1]  # (17,)
        obs_diff = np.mean(np.abs(next_observation - predicted_observation))  # Mean absolute difference

        # Get next action from policy based on next_observation
        next_conditions = {0: next_observation}
        next_action, next_samples, _, _ = policy(next_conditions, batch_size=args.batch_size, verbose=args.verbose)
        predicted_action = actions[0, h+1]  # (6,)
        action_diff = np.mean(np.abs(next_action - predicted_action))  # Mean absolute difference
    else:
        obs_diff = None
        action_diff = None

    # Print comparison
    print(f"h: {h} | Observation diff: {obs_diff if obs_diff is not None else 'N/A'} | Action diff: {action_diff if action_diff is not None else 'N/A'}")

    