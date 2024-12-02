import numpy as np


def walker_env_fn_single(env, param_idx, param):
        
    if param_idx == 0:
        env.unwrapped.model.body_mass[1] = param
    elif param_idx == 1:
        env.unwrapped.model.geom_friction[0,0] = param
    elif param_idx == 2:
        env.unwrapped.model.dof_damping[3] = param
    elif param_idx == 3:
        env.unwrapped.model.dof_damping[6] = param
    else:
        raise ValueError("Invalid param_idx")
    
    return env


