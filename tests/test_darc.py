from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True, cuda_visible_devices=[0])

from diff_trans.envs.gym_wrapper import get_env

num_envs = 10
iterations = 100000
steps_per_iter = 1
updates_per_iter = 1
log_interval = 500

learning_rate = 1e-3

Env = get_env("Reacher-v5")
print("Creating environment...", end="")
env = Env(num_envs=num_envs)
eval_env = Env(num_envs=10)
print(" Done")

from torch.optim import Adam
from baselines.darc.agent import DarcAgent
from baselines.darc.models import ActorNet, QValueNet, ValueNet, ClassifierNet

actor_fc_layers = (256, 256)
q_value_fc_layers = (256, 256)
value_fc_layers = (256, 256)

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor_net = ActorNet(
    observation_dim,
    action_dim,
    hidden_dims=actor_fc_layers,
)
q_value_net = QValueNet(
    observation_dim,
    action_dim,
    hidden_dims=q_value_fc_layers,
)
value_net = ValueNet(
    observation_dim,
    hidden_dims=value_fc_layers,
)
classifier_net = ClassifierNet(observation_dim, action_dim)

# TODO: Check SAC's parameters
darc_agent = DarcAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    actor_net=actor_net,
    q_value_net=q_value_net,
    value_net=value_net,
    classifier_net=classifier_net,
    classifier_optimizer=Adam(classifier_net.parameters(), lr=learning_rate),
    learning_rate=learning_rate,
    # make_actor_optimizer=lambda params: Adam(params, lr=learning_rate),
    # make_q_value_optimizer=lambda params: Adam(params, lr=learning_rate),
    # make_value_optimizer=lambda params: Adam(params, lr=learning_rate),
    # make_classifier_optimizer=lambda params: Adam(params, lr=learning_rate),
    target_update_tau=0.005,
    target_update_period=1,
    # td_errors_loss_fn=td_errors_loss_fn,
    # gamma=gamma,
    # reward_scale_factor=reward_scale_factor,
    # gradient_clipping=gradient_clipping,
    # train_step_counter=global_step,
)

import os
from baselines.darc.agent import predict
import imageio

OUTPUT_DIR = "outputs/test_darc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Creating render environment...", end="")
render_env = Env(num_envs=1, render_mode="rgb_array")
print(" Done")

from baselines.darc.agent import SimpleCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, Bounded

collector = SimpleCollector(env)
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(1000000),
    batch_size=256,
)

print("Collecting initial random experiences")
replay_buffer.extend(
    collector.collect(
        darc_agent.get_init_policy(num_envs), num_steps=300, progress_bar=True
    )
)

from tqdm import tqdm
from baselines.darc.agent import evaluate_policy


def run():
    for i in tqdm(range(iterations)):
        experiences = collector.collect(darc_agent.policy, num_steps=steps_per_iter)
        replay_buffer.extend(experiences)

        for _ in range(updates_per_iter):
            batch = replay_buffer.sample()
            losses = darc_agent.update(batch, None)

        loss_actor = losses["loss_actor"]
        loss_qvalue = losses["loss_qvalue"]
        loss_value = losses["loss_value"]
        if (i + 1) % log_interval == 0:
            print("Updates:", darc_agent.num_updates)
            print("Actor loss:", loss_actor)
            print("Q value loss:", loss_qvalue)
            print("Value loss:", loss_value)

            mean, std = evaluate_policy(eval_env, darc_agent.policy, num_episodes=10)
            print(f"Iteration {i + 1}")
            print(f"Mean: {mean}, Std: {std}")

            darc_agent.save(os.path.join(OUTPUT_DIR, f"iter_{i + 1:06d}.pt"))

            obs = render_env.reset()
            images = [render_env.render(0)]
            for _ in range(100):
                actions = predict(darc_agent.policy, obs)
                obs, _, _, _ = render_env.step(actions)
                images.append(render_env.render(0))

            imageio.mimsave(
                os.path.join(OUTPUT_DIR, f"iter_{i + 1:06d}.gif"),
                images,
                duration=1000 / render_env.metadata["render_fps"],
            )


# with torch.autograd.detect_anomaly():
#     run()
run()
