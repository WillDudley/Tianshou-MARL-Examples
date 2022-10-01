from tianshou.env import PettingZooEnv       # wrapper for PettingZoo environments
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
from pettingzoo.classic import rps_v2  # the Tic-Tac-Toe environment to be wrapped

if __name__ == "__main__":
    env = PettingZooEnv(rps_v2.env())
    policy = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)
    env = DummyVectorEnv([lambda: env])
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=.1)
