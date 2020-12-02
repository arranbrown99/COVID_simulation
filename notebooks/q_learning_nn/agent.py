import numpy as np

class Agent():
    
    def __init__(self, env, learning_rate=0.001):
        self.env = env
        self.n_actions = env.action_space.n
        self.d_states = env.observation_space.shape[0]
        self.learning_rate = learning_rate
        
    def get_action(self, strategy, policy_network, state):
        action = self.get_random_action()
        exploit = False
        if not self.is_learning():
            action = self.predict_action_from_nn(policy_network, state) # exploit
            exploit = True
        else:
            if not strategy.should_explore():
                action = self.predict_action_from_nn(policy_network, state) # exploit
                exploit = True
        return (action, exploit)
    
    def get_num_actions(self):
        return self.n_actions
    
    def get_num_states(self):
        return self.d_states
    
    def get_random_action(self):
        return np.random.choice(self.get_num_actions())
    
    def predict_action_from_nn(self, policy_network, state):
        action = policy_network.predict(state)[0]
        action = np.argmax(action)
        return action
    
    def preprocess_state(self, state):
        return np.reshape(state, [1, self.get_num_states()])
    
    def is_learning(self):
        return self.learning_rate > 0.0