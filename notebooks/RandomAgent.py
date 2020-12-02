import numpy as np

class RandomAgent:

    def __init__(self, env, action_text):
        self.num_of_actions = env.action_space.n
        self.env = env
        self.action_text = action_text
        #print("Agent has " + str(self.num_of_actions) + " actions and will randomly select one at each step")
        
    def get_action(self):
        self.action = np.random.choice(self.num_of_actions)
        return self.action
    
    def get_action_text(self):
        return self.action_text[self.action]
    
    def get_env(self):
        return self.env
    
    def get_chart_title(self):
        return "Random Actions : Stochastic = " + str(self.env.is_stochastic) + " , Noisy = " + str(self.env.is_noisy)
