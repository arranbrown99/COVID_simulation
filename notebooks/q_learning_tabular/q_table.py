import json


class QTable:
    def __init__(self, num_of_actions, q_table=None):
        if q_table:
            self.q_table = q_table
        else:
            self.q_table = {}
        self.num_of_actions = num_of_actions
        
    @staticmethod
    def load_raw_q_table_from_file(filename):
        new_q_table = None
        with open(filename, "r") as file:
            contents = file.read()
            q_table = json.loads(contents)
            new_q_table = {}
            for key, value in q_table.items():
                # Unpacking the json is strictly dependent on saving the json
                # If you update this logic, you must update save_raw_q_table_to_file
                key_as_tuple = tuple(map(int, key.split(', '))) 
                new_q_table[key_as_tuple] = value
        return new_q_table
        
    def save_raw_q_table_to_file(self, filename):
        with open(filename, "w") as file:
            new_q_table = {}
            for key, value in self.q_table.items():
                # Loading the json determines how we unpacking the json later
                # If you update this logic, you must update load_raw_q_table_from_file
                new_key = str(key).strip()[1:-1]
                new_q_table[new_key] = value
            file.write(json.dumps(new_q_table))
            return True
        return False
    
    def get_actions(self, state):
        state_tuple = tuple(state)
        return self.q_table.get(state_tuple, [0]*self.num_of_actions)
    
    def get_action_value(self, state, action_index):
        state_tuple = tuple(state)
        return self.q_table.get(state_tuple, [0]*self.num_of_actions)[action_index]
    
    def set_action_value(self, state, action_index, action_value):
        if not tuple(state) in self.q_table:
            self.q_table[tuple(state)] = [0]*self.num_of_actions
        self.q_table[tuple(state)][action_index] = action_value 