import argparse
import yaml

import cma
import gymnasium as gym
import numpy as np
import torch
import wandb

from utils.replay import ReplayBuffer
from utils.sac import SACAgent


from envs.wrapper import get_env_fn, get_param_fn, get_params_range, sample_params, params_to_array

from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, required=True)

args = parser.parse_args()

with open(args.config, 'r') as f:
    config_file = yaml.safe_load(f)
    
config = config_file['config']

print(config)

simulator = gym.make(config['env_name'])

env_fn = get_env_fn(config['env_name'])

simulator, _ = env_fn(simulator, config['num_params'], config['gt'])

obs_dim = simulator.observation_space.shape[0]
act_dim = simulator.action_space.shape[0]
act_limit = simulator.action_space.high[0]

max_ep_len = simulator.spec.max_episode_steps

ac_kwargs = dict()

agent = SACAgent(obs_dim+config['num_params'], act_dim, act_limit, **ac_kwargs)

agent.ac.pi.load_state_dict(torch.load(config['ckpt'], weights_only=True))

agent.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent.to(device)

wandb.login()
wandb.init(project="sim2real-policy-transfer")

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__ = func.__name__
    return helper

@call_counter
def optim_func(x):
    
    x = np.array(x)
    
    env = simulator
    
    reward = 0
    
    o, _ = env.reset(seed=0)
    
    for i in range(max_ep_len):
        
        a = agent.get_action(np.concatenate([o, x]))
        
        o2, r, d, _, _ = env.step(a)
        
        reward += r
        
        o = o2
        
        if d:
            break
        
    wandb.log({"reward": reward,
            "pred": x[0]})   
        
    return -reward

def run_env(x):
    
    x = np.array(x)
    
    env = simulator
    
    reward = 0
    
    o, _ = env.reset(seed=0)
    
    for i in range(max_ep_len):
        
        a = agent.get_action(np.concatenate([o, x]))
        
        o2, r, d, _, _ = env.step(a)
        
        reward += r
        
        o = o2 
        
        if d:
            break   
        
    return reward

initial_solution = np.array(config['init'])

sigma = 0.5

es = cma.CMAEvolutionStrategy(initial_solution, sigma)

for i in range(50):
    solutions = es.ask()
    es.tell(solutions, [optim_func(x) for x in solutions])
    es.disp()
    print("steps", optim_func.calls, es.result.xbest)

gt_list = []

for i in range(10):
    gt = run_env(es.result.xbest)
    gt_list.append(gt)
    
print("final", sum(gt_list)/len(gt_list))

best_solution = es.result.xbest
best_fitness = es.result.fbest

print("\n最佳解: ", best_solution)
print("最佳適應度（目標函數值）: ", best_fitness)

gt_list = []

for i in range(10):
    gt = run_env(np.array(config['gt']))
    gt_list.append(gt)

print("ground truth", sum(gt_list)/len(gt_list))
