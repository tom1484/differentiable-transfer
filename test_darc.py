from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

from diff_trans.envs.gym import get_env

Env = get_env("InvertedPendulum-v5")
env = Env(num_envs=100)

from baselines.darc.agent import DarcAgent
from baselines.darc.models import ActorNet, QValueNet, ValueNet, ClassifierNet
from torch.optim import Adam

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
classifier_net = ClassifierNet(
    observation_dim,
    action_dim
)

# TODO: Check SAC's parameters
darc_agent = DarcAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    actor_net=actor_net,
    q_value_net=q_value_net,
    value_net=value_net,
    classifier_net=classifier_net,
    actor_optimizer=Adam(actor_net.parameters(), lr=1e-3),
    q_value_optimizer=Adam(q_value_net.parameters(), lr=1e-3),
    value_optimizer=Adam(value_net.parameters(), lr=1e-3),
    classifier_optimizer=Adam(classifier_net.parameters(), lr=1e-3),
    target_update_tau=0.005,
    target_update_period=1,
    # td_errors_loss_fn=td_errors_loss_fn,
    # gamma=gamma,
    # reward_scale_factor=reward_scale_factor,
    # gradient_clipping=gradient_clipping,
    # train_step_counter=global_step,
)

from baselines.darc.agent import SimpleCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

collector = SimpleCollector(env)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(1000000),
    batch_size=256,
)

from tqdm import tqdm
from baselines.darc.agent import evaluate_policy

eval_env = Env(num_envs=100)

for i in tqdm(range(10000)):
    experiences = collector.collect(darc_agent.policy, num_steps=1)
    replay_buffer.extend(experiences)

    batch = replay_buffer.sample()
    darc_agent.update(batch, None)

    if (i + 1) % 100 == 0:
        mean, std = evaluate_policy(eval_env, darc_agent.policy, num_episodes=100)
        print(f"Iteration {i + 1}")
        print(f"Mean: {mean}, Std: {std}")

darc_agent.save("darc_agent.pt")
