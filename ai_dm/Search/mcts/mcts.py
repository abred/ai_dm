from collections.abc import Callable
import logging
from multiprocessing import Process, Queue
import os
import signal
from typing import Optional

from ai_dm.Search.mcts.mcts_node import MCTSNode

logger = logging.getLogger(__name__)


def handler_sig_alrm(signum, frame):
    raise TimeoutError


class MCTS():
    def __init__(
            self,
            set_env_to_state: Callable,
            select_and_expand: Callable, # TreePolicy
            rollout: Callable, # DefaultPolicy
            backprop: Callable,
            best_action: Callable,
            prune_tree: Optional[Callable]
    ):
        self.compute_process = None
        self.set_env_to_state = set_env_to_state
        self.select_and_expand = select_and_expand
        self.rollout = rollout
        self.backprop = backprop
        self.best_action = best_action
        self.prune_tree = prune_tree
        logger.debug("Used set env to state function: %s", set_env_to_state)
        logger.debug("Used select function: %s", select_and_expand)
        logger.debug("Used rollout function: %s", rollout)
        logger.debug("Used backprop function: %s", backprop)
        logger.debug("Used best action function: %s", best_action)
        logger.debug("Used prune tree iteration? %s", prune_tree)

        if os.name == 'nt':
            # 'nt' means Windows, untested
            signal.signal(signal.CTRL_C_EVENT, handler_sig_alrm)
        else:
            signal.signal(signal.SIGALRM, handler_sig_alrm)

    def get_action_iterations(self, state, env, max_iterations):
        return self.__get_action(
            state, env, max_iterations=max_iterations)

    def get_action_time(self, state, env, time_budget, max_iterations=None):
        signal.alarm(time_budget)
        return self.__get_action(state, env, max_iterations=max_iterations)

    def get_action_fork_once(self, state, env, consume_budget, max_iterations=None):
        if self.compute_process is None:
            self.q = Queue()
            self.compute_process = Process(
                target=self.__get_action_queue_loop,
                args=(env, self.q),
                kwargs={"max_iterations": max_iterations})
            self.compute_process.start()
        self.q.put(state)
        consume_budget()
        if os.name == 'nt':
            # 'nt' means Windows, untested
            os.kill(self.compute_process.pid, signal.CTRL_C_EVENT)
        else:
            os.kill(self.compute_process.pid, signal.SIGALRM)
        action = self.q.get()

        return action

    def get_action_fork(self, state, env, consume_budget, max_iterations=None):
        assert self.prune_tree is None, (
            "keeping part of the search tree not possible with this type "
            "of budget computation, new process created after each step")
        q = Queue()
        compute_process = Process(
            target=self.__get_action_queue,
            args=(state, env, q),
            kwargs={"max_iterations": max_iterations})
        compute_process.start()
        consume_budget()
        if os.name == 'nt':
            # 'nt' means Windows, untested
            os.kill(compute_process.pid, signal.CTRL_C_EVENT)
        else:
            os.kill(compute_process.pid, signal.SIGALRM)
        action = q.get()
        compute_process.join()

        return action

    def __get_action_queue_loop(
            self, env, q, max_iterations=None):
        while True:
            state = q.get()
            action = self.__get_action(
                state, env, max_iterations=max_iterations)
            q.put(action)

    def __get_action_queue(self, state, env, q, max_iterations=None):
        action = self.__get_action(
            state, env, max_iterations=max_iterations)
        q.put(action)

    def __get_action(self, state, env, max_iterations=None):
        if self.prune_tree is None or not hasattr(self, "root"):
            self.root = MCTSNode(state, False)
            self.root.count_visited = 1
            logger.info("New root: %s", self.root)
        elif self.prune_tree is not None and hasattr(self, "root"):
            self.prune_tree(self.root, state)

        try:
            it = 0
            while it != max_iterations:
                it += 1
                self.set_env_to_state(env, state)
                node = self.select_and_expand(self.root, env)
                logger.debug("Selected node: %s", node)
                value = self.rollout(node, env)
                logger.debug("Value after rollout: %s", value)
                self.backprop(value, node)
        except TimeoutError:
            print("budget used, returning currently best action")

        # reset env back to root state
        self.set_env_to_state(env, state)
        action = self.best_action(self.root, env.action_space)
        logger.debug(
            "Final selected action after %s iterations: %s", it, action)
        return action
