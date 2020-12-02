class DeterministicAgent:

    def __init__(self, env, action, action_text):
        self.num_of_actions = env.action_space.n
        self.action = action
        self.env = env
        self.action_text = action_text
        #print("Agent has " + str(self.num_of_actions) + " actions and will always choose action " + str(action) + ": " + action_text)
        
    def get_action(self):
        return self.action
    
    def get_action_text(self):
        return self.action_text
    
    def get_env(self):
        return self.env
    
    def get_chart_title(self):
        return "Action = " + self.action_text