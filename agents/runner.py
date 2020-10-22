import virl
import numpy as np
from agents import DeterministicAgent, RandomAgent

from matplotlib import pyplot as plt

env = virl.Epidemic(stochastic=False, noisy=False)
agent = DeterministicAgent(env)
states = []
rewards = []
done = False

s = env.reset()
states.append(s)
while not done:
    s, r, done, i = env.step(action=agent.get_action())
    states.append(s)
    rewards.append(r)

states = np.array(states)
rewards = np.array(rewards)

plt.plot(rewards)

print(states[:, 0][0:4])
print(states[:, 1][0:4])
print(states[:, 2][0:4])
print(states[:, 3][0:4])

print("rewards " + str(rewards[:4]))

print(states.shape)
print(rewards.shape)


total_reward = np.sum(rewards)
print("Total reward: " + str(total_reward))
