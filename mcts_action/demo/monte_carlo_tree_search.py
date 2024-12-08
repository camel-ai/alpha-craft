from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
    """
    Monte Carlo tree searcher. First rollout the tree then choose a move.
    """

    def __init__(self, world_model, vf_agent, exploration_weight=1, max_depth=None, max_expansion=None):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth  # Maximum depth for rollouts
        self.max_expansion = max_expansion  # Maximum number of children to expand
        self.world_model = world_model
        self.vf_agent = vf_agent

    def choose(self, node):
        """
        Choose the best successor of node. (Choose a move in the game)
        """
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        """
        Make the tree one layer better. (Train for one iteration.)
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf, depth=0)
        self._backpropagate(path, reward)

    def _select(self, node):
        """
        Find an unexplored descendent of `node`.
        """
        path = []
        depth = 0
        while True:
            path.append(node)
            if self.max_depth is not None and depth >= self.max_depth:
                return path
            if node not in self.children or not self.children[node]:
                # Node is either unexplored or terminal
                return path
            # children of the node - children in tree, to identify unexplored ones
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper
            depth += 1

    def _expand(self, node):
        """
        Update the `children` dict with the children of `node`.
        Only expand up to `max_expansion` nodes if specified.
        """
        if node in self.children:
            return  # Already expanded
        children = node.find_children()
        if self.max_expansion is not None:
            children = set(list(children)[: self.max_expansion])  # Limit expansions
        self.children[node] = children

    def _simulate(self, node, depth=0):
        """
        Returns the reward for a random simulation (to completion) of `node`.
        Simulations are terminated early if `max_depth` is reached.
        """
        while True:
            if node.is_terminal() or (self.max_depth is not None and depth >= self.max_depth):
                reward = self.vf_agent.eval(node.state['observation'], node.state['action'])
                return reward
            new_node = node.find_random_child(self.world_model)
            """
            To simulate, you take the observation of the current node and the action in the new node
            """
            new_obs = self.world_model.step(node.state['observation'], new_node.state['action'])
            new_node.update_state({'observation': new_obs, 'action': new_node.state['action']})
            new_node.is_terminal = self.is_terminal()
            depth += 1

    def _backpropagate(self, path, reward):
        """
        Send the reward back up to the ancestors of the leaf.
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        """
        Select a child of node, balancing exploration & exploitation.
        """
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            """
            Upper confidence bound for trees.
            """
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
