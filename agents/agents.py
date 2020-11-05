import numpy as np


class RandomAgent:

    def __init__(self, env):
        self.num_of_actions = env.action_space.n
        print(env.action_space)
        print("Agent has " + str(self.num_of_actions) + " actions")

    def get_action(self):
        action = np.random.choice(self.num_of_actions)
        return action

# This is an agent
class DeterministicAgent:

    def __init__(self, env):
        self.num_of_actions = env.action_space.n
        print("Agent has " + str(self.num_of_actions) + " actions")

    def get_action(self):
        action = 0
        return action
