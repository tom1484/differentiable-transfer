import argparse
import itertools
import time
from copy import deepcopy
from typing import Tuple
import yaml

import gymnasium as gym
import numpy as np
import torch

from utils.replay import ReplayBuffer
from utils.sac import SACAgent


from envs.wrapper import get_env_fn, get_param_fn, get_params_range, sample_params, params_to_array

BUFFER_SIZE = 10000
NUM_FRAMES = 10

    
def sac(
        seed=0, 
        replay_size=int(1e6),
        batch_size=100,
        update_every=50,
    ):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='./configs/hc_1.yaml')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_file = yaml.safe_load(f)
        
    config = config_file['config'] 
    
    print(config)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    simulator = gym.make(config['env_name'])

    env_fn = get_env_fn(config['env_name'])
    
    act_dim = simulator.action_space.shape[0]
    obs_dim = simulator.observation_space.shape[0] + config['num_params']
    act_limit = simulator.action_space.high[0]
    max_ep_len = simulator.spec.max_episode_steps

    ac = SACAgent(obs_dim, act_dim, act_limit)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    start_time = time.time()
    ep_ret, ep_len = 0, 0

    test_epochs = []
    
    env_params = get_params_range(config['env_name'], config['num_params'])
    
    for epoch in range(config['epochs']):
        
        params = sample_params(env_params, epoch, config['epochs'])
        
        params_array = params_to_array(params)
        
        o, _ = simulator.reset()
        
        viz_o = np.concatenate([o, params_array])
        
        simulator, _ = env_fn(simulator, config['num_params'], params_array)
        
        ep_ret, ep_len = 0, 0
        
        for timestep in range(max_ep_len):
        
            if epoch == 0:
                a = simulator.action_space.sample()
            else:
                a = ac.get_action(viz_o)
            
            o2, r, d, _, _ = simulator.step(a)
            viz_o2 = np.concatenate([o2, params_array])
            ep_ret += r
            ep_len += 1
            
            d = False if ep_len==max_ep_len else d
            
            replay_buffer.store(viz_o, a, r, viz_o2, d)
            
            viz_o = viz_o2
            
            if (epoch > 0) and (timestep % update_every == 0):
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    ac.update(data=batch)
                    
            if d:
                break
                    

        now = time.time()

        s = (int)(now - start_time)
        test_epochs.append(epoch)
        print(f"Epoch: {epoch}, Reward: {ep_ret:.2f}, Time: {s//3600:02}:{s%3600//60:02}:{s%60:02}", )

    print('Training complete')
    torch.save(ac.ac.pi.state_dict(), f'./ckpt/{config["env_name"]}_{config["num_params"]}.pth')
        

if __name__ == '__main__':
    sac()