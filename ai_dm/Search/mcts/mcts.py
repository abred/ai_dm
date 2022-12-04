from collections.abc import Callable
from copy import deepcopy
import logging

from ai_dm.Search.mcts.mcts_node import MCTSNode

logger = logging.getLogger(__name__)


class MCTS():
    def __init__(
            self,
            budget,
            update_budget: Callable,
            select_and_expand: Callable, # TreePolicy
            rollout: Callable, # DefaultPolicy
            backprop: Callable,
            best_action: Callable,
    ):
        self.budget = budget
        self.update_budget = update_budget
        self.select_and_expand = select_and_expand
        self.rollout = rollout
        self.backprop = backprop
        self.best_action = best_action
        logger.debug("Used update budget function: %s", update_budget)
        logger.debug("Used select function: %s", select_and_expand)
        logger.debug("Used rollout function: %s", rollout)
        logger.debug("Used backprop function: %s", backprop)
        logger.debug("Used best action function: %s", best_action)

    def get_action(self, state, env):
        self.root = MCTSNode(state, False)
        self.root.count_visited = 1
        logger.info("New root: %s", self.root)

        budget_t = deepcopy(self.budget)
        while budget_t:
            env_t = deepcopy(env)
            logger.debug("Budget left: %d", budget_t)
            node = self.select_and_expand(self.root, env_t)
            logger.debug("Selected node: %s", node)
            value = self.rollout(node, env_t)
            logger.debug("Value after rollout: %s", value)
            self.backprop(value, node)
            budget_t = self.update_budget(budget_t)

        action = self.best_action(self.root, env.action_space)
        logger.debug("Final selected action: %s", action)

        return action
