import virl
import numpy as np
from matplotlib import pyplot as plt

def run(agent):
    env = agent.get_env()
    states = []
    rewards = []
    done = False

    s = env.reset()
    states.append(s)
    while not done:
        s, r, done, i = env.step(action=agent.get_action()) # deterministic agent
        states.append(s)
        rewards.append(r)
    return (states, rewards)


def plot(agent, states, rewards):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i])
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].set_title(agent.get_chart_title())
    axes[0].legend()
    axes[1].plot(rewards);
    axes[1].set_title("Total reward = " + str(np.sum(rewards)))
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')