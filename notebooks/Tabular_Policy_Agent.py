import os
import numpy as np
os.chdir('..')
import virl
from matplotlib import pyplot as plt

import pandas as pd
from IPython import display

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
        self.print_out_every_x_episodes = int(self.episodes/50)
        self.times_exploited = 0
        
        
         # hyper parameters for discretising state data
        self.highest = 600000000
        self.lowest = 0
        self.number_bins = 20
        
        # hyper parameters for epsilon explore
        self.initial_epsilon = 1 # initial
        self.decrease_factor = (1/self.episodes)/1.25 # epsilon
        print("Decrease Factor: " + str(self.decrease_factor))
        
    def continous_to_discrete(self,continous_state):
        bins = np.linspace(self.lowest,self.highest,num=self.number_bins)
        discrete = np.digitize(continous_state,bins)
        return tuple(discrete)
    
    def train(self):
        states,all_rewards, all_total_rewards = self.run_all_episodes("Training")
        return states,all_rewards, all_total_rewards

    
    def evaluate(self,episodes=100):
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

#             if episode % self.print_out_every_x_episodes == 0:
#                 print("Episode number: " + str(episode) + ". Total reward in episode: " + str(total_reward) + ". Episode executed with epsilon = " + str(epislon))
#                 print("Average total reward in last " + str(self.print_out_every_x_episodes) + " episodes: " + str(np.mean(all_rewards[-self.print_out_every_x_episodes:])))
#                 print("Times exploited the last episode " + str(self.times_exploited))
#                 print("-----")
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