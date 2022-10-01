import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

def get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320
        )

    if agent_opponent is None:
        agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def get_env():
    return PettingZooEnv(tictactoe_v3.env())


def train_agent(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(10)])
    test_envs = DummyVectorEnv([get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        model_save_path = os.path.join(
            'log', 'rps', 'dqn', 'policy.pth'
        )
        os.makedirs(os.path.join('log', 'rps', 'dqn'), exist_ok=True)
        torch.save(
            policy.policies[agents[1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.6

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # trainer
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric
    )

    return result, policy.policies[agents[1]]

# train the agent and watch its performance in a match!
if __name__ == "__main__":
    result, agent = train_agent()
