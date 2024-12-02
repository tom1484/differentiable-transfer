import argparse
import itertools
import time
import yaml

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.replay import ReplayBuffer, SpBuffer
from utils.sac import SACAgent
from utils.spm_model import SpmModel
from envs.wrapper import get_env_fn, get_param_fn

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
    
    wandb.init(
        entity="differentiable-transfer",
        project="spm-single",
        config=config,
        name=f'{config["env_name"]}_{config["num_params"]}_params',
        tags=[config['env_name'], f'{config["num_params"]}_params'],
    )
    
    real_timesteps = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    simulator = gym.make(config['env_name'])
    
    real_env = gym.make(config['env_name'])
    
    sp_env = gym.make(config['env_name'])
    
    env_fn = get_env_fn(config['env_name'])
    
    real_env, _ = env_fn(
        env=real_env,
        num_params=config['num_params'],
        params=np.array(config['ground_truth']),
    )
    
    param_gt = np.array(config['init_param'])
    
    act_dim = simulator.action_space.shape[0]
    obs_dim = simulator.observation_space.shape[0]
    act_limit = simulator.action_space.high[0]
    max_ep_len = simulator.spec.max_episode_steps

    ac = SACAgent(obs_dim, act_dim, act_limit)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    start_time = time.time()
    ep_ret, ep_len = 0, 0

    test_epochs = []
    
    sp_buffer = SpBuffer(obs_dim, act_dim, config['num_params'], max_ep_len, 10, BUFFER_SIZE)
    
    spm_model = SpmModel(obs_dim, act_dim, NUM_FRAMES, config['num_params'], hidden_dim=256).to(device)
    
    optimizer = torch.optim.Adam(spm_model.parameters(), lr=1e-3)
    
    loss_fn = nn.BCELoss()
    
    for epoch in range(config['epochs']):
        
        print("=" * 50, f"\nEpoch: {epoch}")
    
        for policy_iter in range(config['policy_iters']):
            
            o, _ = simulator.reset()
            
            simulator, _ = env_fn(simulator, config['num_params'], param_gt)
            
            ep_ret, ep_len = 0, 0
            
            for timestep in range(max_ep_len):
            
                if epoch == 0 and policy_iter == 0:
                    a = simulator.action_space.sample()
                else:
                    a = ac.get_action(o)
                
                o2, r, d, _, _ = simulator.step(a)
                ep_ret += r
                ep_len += 1
                
                d = False if ep_len==max_ep_len else d
                
                replay_buffer.store(o, a, r, o2, d)
                
                o = o2
                
                if (policy_iter > 0 or epoch != 0) and timestep % update_every == 0:
                    for j in range(update_every):
                        batch = replay_buffer.sample_batch(batch_size)
                        ac.update(data=batch)
                        

            now = time.time()

            s = (int)(now - start_time)
            test_epochs.append(epoch)
            print(f"Policy_iter: {policy_iter}, Reward: {ep_ret:.2f}, Time: {s//3600:02}:{s%3600//60:02}:{s%60:02}", )

        for _ in range(config['spm_iters']):
            
            o, _ = sp_env.reset()
            
            sp_env, sample_param = env_fn(sp_env, config['num_params'], param_gt, deterministic=False)
            
            obs_data, act_data = np.zeros((max_ep_len, obs_dim)), np.zeros((max_ep_len, act_dim))
            
            for timestep in range(max_ep_len):
                
                a = ac.get_action(o)
                
                obs_data[timestep] = o
                
                act_data[timestep] = a
                
                o2, r, d, _, _ = sp_env.step(a)
                
                o = o2
                
            sp_buffer.store(obs_data, act_data, sample_param)    
        
        for _ in range(config['spm_train_iters']):
            
            data_batch, label_batch = sp_buffer.dataset(128, param_gt, bin=True)
            
            for i in range(len(data_batch)):
                
                data, label = data_batch[i].to(device), label_batch[i].to(device)
                
                pred = spm_model(data)
                
                loss = loss_fn(pred, label)
                
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
            
        
        s = (int)(now - start_time)
        print(f"Spm_iter. Loss: {loss.item():.5f}, Time: {s//3600:02}:{s%3600//60:02}:{s%60:02}", )

        o, _ = real_env.reset()
        
        obs_data, act_data = np.zeros((max_ep_len, obs_dim)), np.zeros((max_ep_len, act_dim))
        
        max_reward = np.NINF
        max_reward_epoch = 0
        
        reward = 0
        
        for timestep in range(max_ep_len):
            
            real_timesteps += 1
            
            a = ac.get_action(o)
            
            obs_data[timestep] = o
            
            act_data[timestep] = a
            
            o2, r, d, _, _ = real_env.step(a)
            
            o = o2
            
            reward += r 
            
        if reward > max_reward:
            max_reward = reward
            max_reward_epoch = epoch
        
        data = np.concatenate([obs_data, act_data], axis=1)
        
        data_input = []
        
        for j in range(max_ep_len-10):
            
            oa = data[j:j+10].reshape(-1)
            
            data_input.append(np.concatenate([oa, param_gt]))
        
        data_input = torch.tensor(np.array(data_input), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred = spm_model(data_input.to(device))
        
        pred = np.mean(pred.detach().cpu().numpy())
        
        print(f"Epoch. Reward: {reward:.2f}")
        
        with torch.no_grad():
            pred = spm_model(data_input).detach().cpu().numpy()

        print(pred.shape)

        pred_avg = np.mean(pred, axis=0)
        
        print(pred_avg.shape)
        
        print(f'Now param={param_gt}, Prediction={pred_avg}')
            
        param_gt = param_gt + 0.5 * (pred_avg - 0.5)
        
        param_gt = np.clip(param_gt, 0.01, None)
            
        print(f"New param: {param_gt}")

        print("=" * 50)
        
        log_data = {
            "steps": real_timesteps,
            "sim_reward": ep_ret,
            "real_reward": reward,
            "mse_loss": np.mean((param_gt - np.array(config['ground_truth'])) ** 2),
        }
        
        for i in range(len(param_gt)):
            log_data[f"pred_param_{i}"] = param_gt[i]
        
        wandb.log(log_data)
        
        if (epoch <= 25) and (epoch % 5 == 0):
            ac.reset()
        
    wandb.finish()

if __name__ == '__main__':
    sac()