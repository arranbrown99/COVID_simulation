from collections import deque, namedtuple
import random
from operator import attrgetter
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'new_state', 'reward'))

class ReplayMemory():
    """
    Implement a replay buffer using the deque collection
    """

    def __init__(self, size, batch_size):
        self.capacity = size
        self.memory = deque(maxlen=size)
        self.batch_size = batch_size

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def pop(self):
        return self.memory.pop()

    def sample(self):
        return random.sample(self.memory, self.batch_size)
    
    def can_sample(self):
        return len(self.memory) >= self.batch_size
    
    def extract_samples(self):
        returned_batch_size = int(self.batch_size/2)
        
        transitions = self.sample()
        best_transitions = sorted(transitions, key=attrgetter("reward"), reverse=True)
        transitions = best_transitions[:returned_batch_size]
        batch = Transition(*zip(*transitions))
        train_rewards = np.array(batch.reward)
        train_states = np.array(batch.state)
        train_new_state = np.array(batch.new_state)
        train_actions = np.array(batch.action)
        return train_rewards, train_states, train_new_state, train_actions, returned_batch_size
    
 