from matplotlib import pyplot as plt
import numpy as np
import itertools
import os

class EvalData():
    
    NUM_OF_PROBLEM_IDS = 10
    
    def __init__(self, agent_name, stochastic, noisy):
        self.agent_name = agent_name
        self.rewards = {}
        self.stochastic = stochastic
        self.noisy = noisy
        
    def add_rewards(self, problem_id, rewards):
        self.rewards[problem_id] = rewards
        
    def create_plot(self):
        fig, axes = plt.subplots(5, 2, constrained_layout=True, figsize=(10,10),
                                 subplot_kw={'ylim':(-2.5, 0), 'xlabel':'Episode', 'ylabel':'Reward', 'yticks':np.arange(-2.5, 0, 0.5)})
        fig.suptitle("Stochastic = " + str(self.stochastic) + ", Noisy = " + str(self.noisy))
        
        num_episodes = len(self.rewards[0])
        for x,y in list(itertools.product(range(0, 5), range(0, 2))):
            axes[x, y].plot(np.arange(1, num_episodes + 1), self.rewards[(2*x) + y])
            axes[x, y].set_title(self.get_title((2*x) + y))
        
        filename = self.get_filename(fig=True)
        plt.savefig(filename)
        print("Saved figure at " + filename)
        
    def create_table(self):
        fig = plt.figure(dpi=80)
        ax = fig.add_subplot(1,1,1)
        ax.set_title("")

        rows = []
        for k, v in self.rewards.items():
            rows.append([k, np.mean(np.array(v)), np.std(np.array(v))])

        all_total_rewards = np.array([row[1] for row in rows])
    
        max_reward = np.max(all_total_rewards)
        min_reward = np.min(all_total_rewards)

        GREEN = "#90ee90"
        RED = "#FF7F7F"

        cellColours = []
        for row in rows:
            reward = row[1]
            if reward == max_reward:
                cellColours.append([GREEN, GREEN, GREEN])
                continue
            if reward == min_reward:
                cellColours.append([RED, RED, RED])
                continue
            cellColours.append(["w","w","w"])
            
        rows = [[row[0], round(row[1], 2), round(row[2], 2)] for row in rows]
        column2title = "Average Reward" if self.stochastic or self.noisy else "Reward"
        table = ax.table(cellText=rows, colLabels=["Problem Id", column2title, "Std. Dev."], cellLoc="center", loc='center', cellColours=cellColours)
        table.set_fontsize(11)
        table.scale(1,1.6)
        ax.axis('off')
        
        filename = self.get_filename(fig=False)
        plt.savefig(filename)
        print("Saved table at " + filename)
          
            
    def get_title(self, problem_id):
        return self.agent_name + " problem id " + str(problem_id)
    
    def get_filename(self, fig):
        filename = "eval_output" + os.path.sep + self.agent_name
        if self.stochastic:
            filename += " Stochastic"
        if self.noisy:
            filename += " Noisy"
        if fig:
            filename += " Figure.png"
        else:
            filename += " Table.png"
        return filename

    def print_average_reward(self):
        for k,v in self.rewards.items():
            problem_id = str(k)
            reward = np.mean(np.array(v))
            print("ID: " + problem_id + " Reward: " + str(reward))
            
def create_full_table_for_deterministic(all_eval_data, stochastic, noisy):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Total Reward for each Deterministic Action")
    
    no_itervention_values = []
    for v in all_eval_data[0].rewards.values():
        no_itervention_values.append((np.mean(np.array(v))))

    full_lockdown_values = []
    for v in all_eval_data[1].rewards.values():
        full_lockdown_values.append((np.mean(np.array(v))))

    track_trace_values = []
    for v in all_eval_data[2].rewards.values():
        track_trace_values.append((np.mean(np.array(v))))

    social_distancing_values = []
    for v in all_eval_data[3].rewards.values():
        social_distancing_values.append((np.mean(np.array(v))))

    rows = []
    for i in range(10):        
        rows.append([i, no_itervention_values[i], full_lockdown_values[i], track_trace_values[i], social_distancing_values[i]])

    rows = [[row[0], round(row[1], 2), round(row[2], 2), round(row[3], 2), round(row[4], 2)] for row in rows]
    table = ax.table(cellText=rows, colLabels=["Problem Id",
                                               "No Intervention",
                                               "Full Lockdown",
                                               "Track & Trace", 
                                               "Social Distancing"],
                     cellLoc="center", loc='center')
    table.set_fontsize(14)
    table.scale(1,1.6)
    ax.axis('off')

    filename = "eval_output" + os.path.sep + "Deterministic"
    if stochastic:
        filename += " Stochastic"
    if noisy:
        filename += " Noisy"
    filename += " Table.png"
    plt.savefig(filename)
    print("Saved table at " + filename)