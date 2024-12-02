import numpy as np

def halfcheetah_env_fn(env, num_params, params, deterministic=True):
    
    assert len(params) == num_params
    assert len(params) == 1 or len(params) == 2 or len(params) == 4
    
    if not deterministic:
        params = np.random.normal(params, params/2)
        params = np.clip(params, 0, None)
    
    for i in range(num_params):
        assert params[i] >= 0
    
    if num_params > 0:
        env.unwrapped.model.body_mass[1] = params[0]
    if num_params > 1:
        env.unwrapped.model.geom_friction[0,0] = params[1]
    if num_params > 2:
        env.unwrapped.model.dof_damping[3] = params[2]
    if num_params > 3:
        env.unwrapped.model.dof_damping[6] = params[3]
    return env, params

def halfcheetah_get_params(env, num_params):
    
    param_list = []
    
    if num_params > 0:
        param_list.append(env.unwrapped.model.body_mass[1])
    if num_params > 1:
        param_list.append(env.unwrapped.model.geom_friction[0,0])
    if num_params > 2:
        param_list.append(env.unwrapped_model.dof_damping[3])
    if num_params > 3:
        param_list.append(env.unwrapped_model.dof_damping[6])
    
    return np.array(param_list)

halfcheetah_params = {
    'torso_mass': [0, 20, 6.25],
    'friction': [0.001, 1, 0.4],
    'f_armature': [0.01, 0.4, 0.1],
    'b_armature': [0.01, 0.4, 0.1],
}

