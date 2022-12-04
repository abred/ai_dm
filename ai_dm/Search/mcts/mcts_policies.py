from collections.abc import Callable
import logging

from gym.spaces import Discrete
import numpy as np

from ai_dm.Search.mcts.mcts_node import MCTSNode

logger = logging.getLogger(__name__)


def make_tuple(val):
    if isinstance(val, tuple):
        return val
    elif isinstance(val, list):
        return tuple(val)
    elif isinstance(val, np.ndarray):
        return tuple(list(val))
    else:
        return (val,)


def uct_action_selection(*, exploration_constant, lower_bound_reward, upper_bound_reward):

    exploration_constant = exploration_constant
    lbr = lower_bound_reward
    ubr = upper_bound_reward

    def action_selection(node, action_space):
        logger.debug("UCT: node %s, action space: %s", node, action_space)
        assert isinstance(action_space, (Discrete, list)), (
            "this version of UCT is only implemented for Discrete action spaces")
        max_value = -np.inf
        selected_action = None

        exploration_factor = np.sqrt(2) * exploration_constant
        if isinstance(action_space, Discrete):
            actions = range(action_space.start, action_space.start + action_space.n)
        else:
            actions = action_space

        for action in actions:
            if not node.count_action_tried.get(action):
                value = np.inf
                value_q = None
                value_u = None
                value_q_orig = None
            else:
                value_q_orig = node.value_action_tried[action]
                value_q = (value_q_orig - lbr)/(ubr - lbr)
                value_u = (np.sqrt(np.log(node.count_visited)/
                                   node.count_action_tried[action]))
                value = value_q + exploration_factor * value_u

            logger.debug("Action %s, value: %s (q: %s, u: %s, orig: %s)",
                         action, value, value_q, value_u, value_q_orig)
            if value > max_value:
                max_value = value
                selected_action = action

        logger.debug("Selected action %s with max value %s",
                     selected_action, max_value)
        return selected_action

    return action_selection


def random_action_selection(state, action_space):
    action = action_space.sample()
    logger.debug("Random: state %s, selected %s",
                 state, action)
    return action


def standard_tree_policy(
        select_action: Callable,
):

    select_action_func = select_action

    def apply_tree_policy(node, env):
        logger.debug("Applying standard tree policy to node: %s", node)
        done = False
        while not done and node.count_visited != 0:
            action = select_action_func(node, env.action_space)
            logger.debug("Selected action %s", action)
            res = env.step(action)
            succ_state, reward, done = res[:3]
            succ_state = make_tuple(succ_state)
            succ_node = node.children.setdefault(
                (action, succ_state),
                MCTSNode(
                    succ_state, done, parent=node, action=action,
                    reward=reward))
            logger.debug("Transitioned to node %s", succ_node)
            node = succ_node

        return node

    return apply_tree_policy


def standard_default_policy(*, max_depth):

    def default_policy(node, env):
        state = node.state
        value = node.cum_reward
        logger.debug("Doing rollout with standard default policy")

        if node.terminal:
            return value

        depth = 1
        done = False
        while not done:
            depth += 1
            action = random_action_selection(state, env.action_space)
            res = env.step(action)
            succ_state, reward, done = res[:3]
            value += reward
            state = succ_state
            if depth > max_depth:
                value -= max_depth
                break

        return value

    return default_policy


def standard_backprop(value, node):
    logger.debug("Doing standard backprop")
    while node:
        node.seen_values.append(value)
        node.count_visited += 1
        if node.parent:
            node.parent.count_action_tried[node.action] = (
                node.parent.count_action_tried.get(node.action, 0) + 1)
            node.parent.value_action_tried[node.action] = np.sum(
                node.seen_values)/node.count_visited

        node = node.parent


def standard_best_action_selection(node, action_space):
    logger.debug("Possible actions applicable during standard best action "
                 "selection: %s", action_space)
    assert isinstance(action_space, Discrete), (
        "this version of action selection is only implemented for Discrete action spaces")
    logger.debug("Children of current node: %s", node.children)
    action_counts = np.zeros(action_space.n, dtype=np.float32)

    for child in node.children.values():
        action_id = child.action - action_space.start
        action_counts[action_id] += child.count_visited

    logger.info("Action counts: %s", action_counts)
    action = action_space.start + np.argmax(action_counts)
    return action
