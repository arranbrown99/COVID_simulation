from matplotlib import pyplot as plt
import numpy as np

def plot(state,agent, total_rewards, rewards,title):
    fig, axes = plt.subplots(1, 4, figsize=(40, 16))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(state)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);
        
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[0].title.set_text('Final state')
    
   
    axes[1].plot(total_rewards);
    axes[1].set_xlabel('episode number')
    axes[1].set_ylabel('total reward r(t)')
    axes[1].title.set_text('Total reward per episode')


    axes[2].plot(rewards[0]);
    axes[2].set_xlabel('episode number')
    axes[2].set_ylabel('total reward r(t)')
    axes[2].title.set_text('First Episode reward')
    
   
    axes[3].plot(rewards[-1]);
    axes[3].set_xlabel('episode number')
    axes[3].set_ylabel('total reward r(t)')
    axes[3].title.set_text('Final Episode reward')
    
    fig.suptitle(title) 
