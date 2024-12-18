import numpy as np
from random import choice
import agent.actions as actions
# import copy

class Node:
    def __init__(self, state, num_actions=None, env_state=None, parent=None, question=None):
        self.state = {'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.num_actions = num_actions
        # self.children = [] 
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        # self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.env_state = env_state

    def generate_children_action(self, children, action_key, action_value_max):
       new_children = []
       for i in range(action_value_max):
            for child in children:
                child_action = {**child.state['action']} ## copy dict to generate new actions
                child_action[action_key] = str(i) ## Format for ActionInterface input dict values
                new_children.append(Node(state={'observation': '', 'action': child_action}, parent = self))
       return new_children
        
    def find_children(self):
        action_dict = {
            "forward_backward_action": "0",
            "move_left_right_action": "0",
            "jump_sneak_sprint_action": "0",
        }
        # action_space = actions.ActionSpace(action_dict)
        children = [Node(state={'observation': '', 'action': action_dict}, parent = self)]       

        children = self.generate_children_action(children, "forward_backward_action", 3)
        children = self.generate_children_action(children, "move_left_right_action", 3)
        children = self.generate_children_action(children, "jump_sneak_sprint_action", 4)
        self.children = children    # Stopgap measure to avoid infinite loop
        return children

        

    def find_random_child(self):
        """
        simulator: a simulator, assuming there is a terminal flag
        filters:  allows removing certain actions by indices per node. Not used at the moment
        """
        # if simulator.is_terminal():
        #     return None
        print("agent_node.py:find_random_child len(self.children) ", len(self.find_children()))
        return choice(self.children)
        
    # To be done post simulation
    def update_state(self, new_state_dict):
        self.state = new_state_dict

    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    # @property
    # def is_terminal(self):
    #     return self.is_terminal

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}" #, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            # 'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }