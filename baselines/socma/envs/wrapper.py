from itertools import islice

import numpy as np

from envs.halfcheetah import halfcheetah_env_fn, halfcheetah_get_params, halfcheetah_params
from envs.reacher import reacher_env_fn, reacher_get_params
from envs.walker import walker_env_fn_single

def get_env_fn(env_name):
    if env_name == 'HalfCheetah-v4':
        return halfcheetah_env_fn
    elif env_name == 'Reacher-v4':
        return reacher_env_fn
    
def get_param_fn(env_name):
    if env_name == 'HalfCheetah-v4':
        return halfcheetah_get_params
    elif env_name == 'Reacher-v4':
        return reacher_get_params
    
def get_params_range(env_name, num_params):
    if env_name == 'HalfCheetah-v4':
        params = halfcheetah_params
        sliced_params = dict(islice(params.items(), num_params))
        return sliced_params
    
def sample_params(env_params_range, epoch, epochs):
    
    env_params_sample = {}
    
    for key, value in env_params_range.items():
        if epoch <= epochs/2:
            lower = value[0] * (epoch / (epochs/2)) + value[2] * (1 - epoch / (epochs/2))
            upper = value[1] * (epoch / (epochs/2)) + value[2] * (1 - epoch / (epochs/2))
            env_params_sample[key] = np.random.uniform(low=lower, high=upper)
        else:
            env_params_sample[key] = np.random.uniform(low=value[0], high=value[1])
            
    return env_params_sample

def params_to_array(env_params):
        
    params = []
    
    for key, value in env_params.items():
        if key != 'param_num':
            params.append(value)
    
    return np.array(params)

def get_env_fn_single(env_name):
    if env_name == 'Walker2d-v4':
        return walker_env_fn_single
    else:
        raise ValueError("Invalid env_name")