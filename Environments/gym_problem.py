__author__ = 'sarah'

from AI_agents.Search.problem import Problem
import AI_agents.Search.utils as utils

class GymProblem(Problem):

    """Problem superclass
       supporting COMPLETE
    """
    def __init__(self, env, init_state, constraints=[]):
        super().__init__(init_state, constraints)
        self.env = env
        self.counter = 0

    # get the actions that can be applied at the current node
    def get_applicable_actions(self, node):
        action_list = self.env.P[node.state.get_key()].keys()
        return action_list

    # get (all) succesor states of an action and their
    def get_successors(self, action, node):

        #action_list = self.env.P[node.state.__repr__()]
        successor_nodes = []
        transitions = self.env.P[node.state.__str__()][action]
        action_cost = self.get_action_cost(action, node.state)
        for prob, next_state_key, reward, done in transitions:
            info={}
            info['prob'] = prob
            info['reward'] = reward
            next_state = utils.State(next_state_key, done)
            successor_node = utils.Node (state=next_state, parent=node, action=action, path_cost=node.path_cost + action_cost, info=info)
            successor_nodes.append(successor_node)

        return successor_nodes

    def get_action_cost(self, action, state):
        return 1

    def is_goal_state(self, state):
        if state.is_terminal:
            return True
        else:
            return False

    def apply_action(self, action):
        state, reward, done, info = self.env.step(int(action))
        return [state, reward, done, info]





