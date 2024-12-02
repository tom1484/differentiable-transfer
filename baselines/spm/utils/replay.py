from typing import Tuple

import torch
import numpy as np

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class SpBuffer:
    def __init__(self, 
                obs_dim: int, 
                act_dim: int, 
                param_dim: int, 
                traj_len: int,
                window_size: int, 
                buffer_size: int) -> None:
        
        self.data = np.zeros((buffer_size, traj_len, obs_dim + act_dim), dtype=np.float32)
        self.label = np.zeros((buffer_size, param_dim), dtype=np.float32)
        self.traj_len = traj_len
        self.window_size = window_size
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
    
    def store(self, obs_data: np.ndarray, act_data: np.ndarray, param: np.ndarray) -> None:
        
        traj = np.concatenate([obs_data, act_data], axis=1)
        
        self.data[self.ptr] = traj
        self.label[self.ptr] = param
        
        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size
        
    def sample(self, param_gt: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:

        idxs = np.random.randint(0, self.traj_len-self.window_size+1, self.size)
        
        data = np.array([row[start:start+self.window_size] for row, start in zip(self.data[:self.size], idxs)])

        label = self.label[:self.size]
        
        data = data.reshape(data.shape[0], -1)
        
        data = np.concatenate([data, np.tile(param_gt, (data.shape[0], 1))], axis=1)
        
        label = (label > param_gt).astype(np.float32)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32) 
    
    def sample_nonbin(self, param_gt: np.ndarray) -> Tuple[torch.tensor, torch.tensor]:

        idxs = np.random.randint(0, self.traj_len-self.window_size+1, self.size)
        
        data = np.array([row[start:start+self.window_size] for row, start in zip(self.data[:self.size], idxs)])

        label = self.label[:self.size]
        
        data = data.reshape(data.shape[0], -1)
        
        data = np.concatenate([data, np.tile(param_gt, (data.shape[0], 1))], axis=1)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def dataset(self, batch_size: int, param_gt: np.ndarray, bin: bool) -> Tuple[torch.tensor, torch.tensor]:
        
        if bin:
            data, label = self.sample(param_gt)
        else:
            data, label = self.sample_nonbin(param_gt)
        
        data_batch, label_batch = data.split(batch_size), label.split(batch_size)
        
        return data_batch, label_batch
    
class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    