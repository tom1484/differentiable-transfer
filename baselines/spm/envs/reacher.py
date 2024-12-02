import numpy as np

def reacher_env_fn(env, num_params, params, deterministic=True):
    
    assert len(params) == num_params
    assert len(params) == 1 or len(params) == 2
    
    if not deterministic:
        params = np.random.normal(params, params/2)
        params = np.clip(params, 0.05, None)
    
    for i in range(num_params):
        assert params[i] >= 0
    
    if num_params > 0:
        env.unwrapped.model.dof_armature[0] = params[0]
        env.unwrapped.model.dof_armature[1] = params[0]
    if num_params > 1:
        env.unwrapped.model.dof_damping[0] = params[1]
        env.unwrapped.model.dof_damping[1] = params[1]
        
    return env, params

def reacher_get_params(env, num_params):
    
    param_list = []
    
    if num_params > 0:
        param_list.append(env.unwrapped.model.dof_armature[0])
    if num_params > 1:
        param_list.append(env.unwrapped.model.dof_damping[0])
    
    return np.array(param_list)