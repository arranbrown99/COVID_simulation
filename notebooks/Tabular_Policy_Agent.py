import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from IPython import display

import json

def smooth_plot(all_rewards, smoothed_rewards,title): 
    plt.figure(2, figsize=(12, 6))
    plt.clf()
    plt.title(title)
    plt.xlabel("Epsiode") 
    plt.ylabel("Total Reward")
    plt.plot(all_rewards, '--', alpha=0.5) 
    plt.plot(smoothed_rewards) 
    plt.legend(["Rewards", "Rewards (Smoothed)"]) 

class Tabular_Policy_Agent:
        
    def __init__(self,env):
        
        #Hyper parameters
        self.policy_table = {}
        
        self.num_of_actions = env.action_space.n
        self.env = env
        
        self.episodes = 500
        self.times_exploited = 0
        
        
         # hyper parameters for discretising state data
        self.highest = 600000000
        self.lowest = 0
        self.number_bins = 20
        
        # hyper parameters for epsilon explore
        self.initial_epsilon = 1 # initial
        self.decrease_factor = (1/self.episodes)/1.25 # epsilon
        
        
    
    def load_raw_table_from_file(self,filename):
        new_table = None
        with open(filename, "r") as file:
            contents = file.read()
            table = json.loads(contents)
            new_table = {}
            for key, value in table.items():
                # Unpacking the json is strictly dependent on saving the json
                # If you update this logic, you must update save_raw_table_to_file
                key_as_tuple = tuple(map(int, key.split(', '))) 
                new_table[key_as_tuple] = value
        self.policy_table = new_table
        
    def save_raw_table_to_file(self, filename):
        with open(filename, "w") as file:
            new_table = {}
            for key, value in self.policy_table.items():
                # Loading the json determines how we unpacking the json later
                # If you update this logic, you must update load_raw_table_from_file
                new_key = str(key).strip()[1:-1]
                new_table[new_key] = value
            file.write(json.dumps(new_table))
            return True
        return False
        
    def continous_to_discrete(self,continous_state):
        bins = np.linspace(self.lowest,self.highest,num=self.number_bins)
        discrete = np.digitize(continous_state,bins)
        return tuple(discrete)
    
    def train(self):
        states,all_rewards, all_total_rewards = self.run_all_episodes("Training")
        return states,all_rewards, all_total_rewards

    
    def evaluate(self,filename,episodes=100):
        self.load_raw_table_from_file(filename)
        self.episodes = episodes
        self.epsilon = -1000
        states,all_rewards, all_total_rewards = self.run_all_episodes("Evaluation")
        return states,all_rewards, all_total_rewards
    
    
    def run_all_episodes(self,title):
        all_rewards = []
        epislon = self.initial_epsilon # at the start only explore
        all_total_rewards = []
        
        
        for episode in range(1, self.episodes + 1):
            states,rewards = self.run_episode(epislon)
            total_reward = np.sum(rewards)

            self.times_exploited = 0
            all_total_rewards.append(total_reward)
            all_rewards.append(rewards)
            epislon -= self.decrease_factor #hyperparameter
            
        window_size = int(self.episodes/10)
        smoothed_rewards = pd.Series(all_total_rewards).rolling(window_size, min_periods=window_size).mean() 
        this_smoothed_reward = smoothed_rewards.values[-1]
        smooth_plot(all_total_rewards, smoothed_rewards,title)
        
        return states,all_rewards, all_total_rewards
    
    def run_episode(self,epislon):
        rewards = []
        states = []
        actions = []
        done = False
        
        state = self.env.reset()
        states.append(state)
      
        while not done:
            random_number = np.random.random()
            if random_number < epislon:
                #explore
                action = np.random.choice(self.num_of_actions)
                
            else:
                #exploit
                action = self.get_action(state)
                self.times_exploited += 1
              
            new_state, reward, done, i = self.env.step(action=action)
     
            states.append(new_state)
            actions.append(action)    
            rewards.append(reward)
            
            #update policy function
            self.update(new_state,action,reward)
        
            
            state = new_state
        return states,rewards
        
    def update(self,state,action,reward):
        #update the policy table
        state = self.continous_to_discrete(state)
        previous_action,current_best_reward =  self.policy_table.get(state,(-1,-100))
        if reward > current_best_reward:
            self.policy_table[state] = (action,reward)
                                                     
        
    def get_action(self,state):
        #tabular get best action from the policy
        state = self.continous_to_discrete(state)
        action,best_reward = self.policy_table[state]
        return action
    
    def get_action_text(self):
        return action_text
    
    def get_env(self):
        return env
    
    def get_chart_title(self):
        return "Action = " + action_text