from experiments.env import set_env_vars

set_env_vars(jax_debug_nans=True)

import os
import time

import torch
import numpy as np

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from .models import ActorNet, QValueNet, ValueNet, ClassifierNet
from .agent import DarcAgent, SimpleCollector

from diff_trans.envs.gym_wrapper import get_env


def main(
    # root_dir,
    num_envs: int = 128,
    environment_name: str = "InvertedPendulum-v5",
    num_timesteps: int = 1000000,
    initial_collect_steps=10000,
    real_initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    real_collect_interval=10,
    train_steps_per_iteration=10,
    batch_size=1024,
    # critic_obs_fc_layers=None,
    # critic_action_fc_layers=None,
    # critic_joint_fc_layers=(256, 256),
    actor_fc_layers=(256, 256),
    q_value_fc_layers=(256, 256),
    value_fc_layers=(256, 256),
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    classifier_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    # td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    soft_update_interval=1,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    # train_checkpoint_interval=10000,
    # policy_checkpoint_interval=5000,
    # rb_checkpoint_interval=50000,
    log_interval=1000,
    # summary_interval=1000,
    # summaries_flush_secs=10,
    # debug_summaries=True,
    # summarize_grads_and_vars=False,
    train_on_real=False,
    delta_r_warmup=0,
    random_seed=0,
    # checkpoint_dir=None,
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    Env = get_env(environment_name)
    env = Env()
    real_env = Env()

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
        make_actor_optimizer=None,
        make_q_value_optimizer=None,
        make_value_optimizer=None,
        make_classifier_optimizer=None,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        # td_errors_loss_fn=td_errors_loss_fn,
        # reward_scale_factor=reward_scale_factor,
        # gradient_clipping=gradient_clipping,
        # train_step_counter=global_step,
    )

    # Make the replay buffer.
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=replay_buffer_capacity),
        batch_size=batch_size,
    )
    real_replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=replay_buffer_capacity),
        batch_size=batch_size,
    )

    # Create policyies
    collect_policy = darc_agent.policy
    init_collect_policy = darc_agent.random_policy

    # Create collectors
    collector = SimpleCollector(env)
    real_collector = SimpleCollector(real_env)

    # Collect initial replay data.
    replay_buffer.extend(
        collector.collect(init_collect_policy, initial_collect_steps)
    )
    real_replay_buffer.extend(
        real_collector.collect(init_collect_policy, real_initial_collect_steps)
    )

    dataset = iter(replay_buffer)
    real_dataset = iter(real_replay_buffer)

    time_step = None
    real_time_step = None

    def train_step():
        batch = next(dataset)
        real_batch, _ = next(real_dataset)
        return darc_agent.update(batch, real_batch)

    # for _ in range(num_iterations):
    #     start_time = time.time()
    #     time_step, policy_state = collect_driver.run(
    #         time_step=time_step,
    #         policy_state=policy_state,
    #     )
    #     assert not policy_state  # We expect policy_state == ().
    #     if (
    #         global_step.numpy() % real_collect_interval == 0
    #         and global_step.numpy() >= delta_r_warmup
    #     ):
    #         real_time_step, policy_state = real_collect_driver.run(
    #             time_step=real_time_step,
    #             policy_state=policy_state,
    #         )

    #     for _ in range(train_steps_per_iteration):
    #         train_loss = train_step()
    #     time_acc += time.time() - start_time

    #     global_step_val = global_step.numpy()

    #     if global_step_val % log_interval == 0:
    #         logging.info("step = %d, loss = %f", global_step_val, train_loss.loss)
    #         steps_per_sec = (global_step_val - timed_at_step) / time_acc
    #         logging.info("%.3f steps/sec", steps_per_sec)
    #         tf.compat.v2.summary.scalar(
    #             name="global_steps_per_sec", data=steps_per_sec, step=global_step
    #         )
    #         timed_at_step = global_step_val
    #         time_acc = 0

    #     for train_metric in sim_train_metrics:
    #         train_metric.tf_summaries(
    #             train_step=global_step, step_metrics=sim_train_metrics[:2]
    #         )
    #     for train_metric in real_train_metrics:
    #         train_metric.tf_summaries(
    #             train_step=global_step, step_metrics=real_train_metrics[:2]
    #         )

    #     if global_step_val % eval_interval == 0:
    #         for eval_name, eval_env, eval_metrics in zip(
    #             eval_name_list, eval_env_list, eval_metrics_list
    #         ):
    #             metric_utils.eager_compute(
    #                 eval_metrics,
    #                 eval_env,
    #                 eval_policy,
    #                 num_episodes=num_eval_episodes,
    #                 train_step=global_step,
    #                 summary_writer=eval_summary_writer,
    #                 summary_prefix="Metrics-%s" % eval_name,
    #             )
    #             metric_utils.log_metrics(eval_metrics)

    #     if global_step_val % train_checkpoint_interval == 0:
    #         train_checkpointer.save(global_step=global_step_val)

    #     if global_step_val % policy_checkpoint_interval == 0:
    #         policy_checkpointer.save(global_step=global_step_val)

    #     if global_step_val % rb_checkpoint_interval == 0:
    #         rb_checkpointer.save(global_step=global_step_val)
    # return train_loss


if __name__ == "__main__":
    main()
