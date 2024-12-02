import numpy as np

def halfcheetah_env_fn(env, params_idx, params, deterministic=True):
    
    num_params = len(params_idx)
    
    assert len(params) == num_params
    assert len(params) == 1 or len(params) == 2 or len(params) == 4
    
    if not deterministic:
        params = np.random.normal(params, params/2)
        params = np.clip(params, 0, None)
    
    if 0 in params_idx:
        env.unwrapped.model.geom_friction[0,0] = params[params_idx.index(0)]
    if 1 in params_idx:
        env.unwrapped.model.dof_armature[3] = params[params_idx.index(1)]
    if 2 in params_idx:
        env.unwrapped.model.dof_armature[4] = params[params_idx.index(2)]
    if 3 in params_idx:
        env.unwrapped.model.dof_armature[5] = params[params_idx.index(3)]
    if 4 in params_idx:
        env.unwrapped.model.dof_armature[6] = params[params_idx.index(4)]
    if 5 in params_idx:
        env.unwrapped.model.dof_armature[7] = params[params_idx.index(5)]
    if 6 in params_idx:
        env.unwrapped.model.dof_armature[8] = params[params_idx.index(6)]
    if 7 in params_idx:
        env.unwrapped.model.dof_damage[3] = params[params_idx.index(7)]
    if 8 in params_idx:
        env.unwrapped.model.dof_damage[4] = params[params_idx.index(8)]
    if 9 in params_idx:
        env.unwrapped.model.dof_damage[5] = params[params_idx.index(9)]
    if 10 in params_idx:
        env.unwrapped.model.dof_damage[6] = params[params_idx.index(10)]
    if 11 in params_idx:
        env.unwrapped.model.dof_damage[7] = params[params_idx.index(11)]
    if 12 in params_idx:
        env.unwrapped.model.dof_damage[8] = params[params_idx.index(12)]
    if 13 in params_idx:
        env.unwrapped.model.body_mass[1] = params[params_idx.index(13)]
    if 14 in params_idx:
        env.unwrapped.model.body_mass[2] = params[params_idx.index(14)]
    if 15 in params_idx:
        env.unwrapped.model.body_mass[3] = params[params_idx.index(15)]
    if 16 in params_idx:
        env.unwrapped.model.body_mass[4] = params[params_idx.index(16)]
    if 17 in params_idx:
        env.unwrapped.model.body_mass[5] = params[params_idx.index(17)]
    if 18 in params_idx:
        env.unwrapped.model.body_mass[6] = params[params_idx.index(18)]
    if 19 in params_idx:
        env.unwrapped.model.body_mass[7] = params[params_idx.index(19)]
    
    return env, params

def halfcheetah_get_params(env, params_idx):
    
    param_list = []
    
    if 0 in params_idx:
        param_list.append(env.unwrapped.model.geom_friction[0,0])
    if 1 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[3])
    if 2 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[4])
    if 3 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[5])
    if 4 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[6])
    if 5 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[7])
    if 6 in params_idx:
        param_list.append(env.unwrapped.model.dof_armature[8])
    if 7 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[3])
    if 8 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[4])
    if 9 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[5])
    if 10 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[6])
    if 11 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[7])
    if 12 in params_idx:
        param_list.append(env.unwrapped.model.dof_damping[8])
    if 13 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[1])
    if 14 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[2])
    if 15 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[3])
    if 16 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[4])
    if 17 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[5])
    if 18 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[6])
    if 19 in params_idx:
        param_list.append(env.unwrapped.model.body_mass[7])
    
    return np.array(param_list)

halfcheetah_params = {
    'torso_mass': [0, 20, 6.25],
    'friction': [0.001, 1, 0.4],
    'f_armature': [0.01, 0.4, 0.1],
    'b_armature': [0.01, 0.4, 0.1],
}

