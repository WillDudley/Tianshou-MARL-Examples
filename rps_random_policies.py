"""
This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.
"""

from tianshou.env import PettingZooEnv
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
from pettingzoo.classic import rps_v2

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = rps_v2.env()

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(rps_v2.env())

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=.1)
