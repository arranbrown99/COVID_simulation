import numpy as np

class Strategy():
    
    def __init__(self, epsilon, epsilon_decay):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
    def episode_complete(self):
        self.epsilon *= self.epsilon_decay
        
    def should_explore(self):
        return np.random.random() < self.epsilon