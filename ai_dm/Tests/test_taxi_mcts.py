import logging

import gym
# from ai_dm.Environments.gym_envs.gym_problem import GymProblem

import ai_dm.Search.mcts.mcts_policies as mp
from ai_dm.Search.mcts import mcts_factory

logger = logging.getLogger(__name__)


def set_taxi_env_to_state(env, state):
    env.unwrapped.s = state


def main_taxi_mcts():

    # define the environment
    env = gym.make("Taxi-v3", render_mode='ansi').env
    init_state = env.reset()[0]

    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(init_state)
    print(taxi_row, taxi_col, pass_idx, dest_idx)

    print("initial state:", init_state)
    print(env.render())

    budget = 2
    max_depth = 100
    exploration_constant = 0.5
    lower_bound_reward = -10 * max_depth
    upper_bound_reward = 20

    select_and_expand = mp.standard_tree_policy(
        mp.uct_action_selection(
                exploration_constant=exploration_constant,
                lower_bound_reward=lower_bound_reward,
                upper_bound_reward=upper_bound_reward)
    )
    rollout = mp.standard_default_policy(max_depth=max_depth)
    backprop = mp.standard_backprop
    best_action = mp.standard_best_action_selection
    set_env_to_state = set_taxi_env_to_state
    prune_tree = mp.standard_prune_tree_function

    # perform MCTS
    mcts = mcts_factory(
        set_env_to_state,
        select_and_expand,
        rollout,
        backprop,
        best_action,
        prune_tree=prune_tree
    )()

    state = init_state
    terminated = False
    max_iterations = 100
    max_iterations = None

    type_of_budget_comp = "time" # one of (iterations, time, fork, fork_once)
    # type_of_budget_comp = "fork_once"
    while not terminated:
        if type_of_budget_comp == "time":
            action = mcts.get_action_time(
                state, env, budget, max_iterations=max_iterations)
        elif type_of_budget_comp == "iterations":
            action = mcts.get_action_iterations(
                state, env, max_iterations=max_iterations)
        elif type_of_budget_comp == "fork":
            action = mcts.get_action_fork(
                state, env, mp.sleep_for_time(budget),
                max_iterations=max_iterations)
        elif type_of_budget_comp == "fork_once":
            action = mcts.get_action_fork_once(
                state, env, mp.sleep_for_time(budget),
                max_iterations=max_iterations)
        else:
            raise RuntimeError(
                "invalid budget computation type has to be one of "
                "(\"iterations\", \"time\", \"fork\", \"fork_once\")")

        logger.info("Applying action: %s", action)
        if action is None:
            env.render()
            break
        state, _, terminated, _, _ = env.step(action)

        print(env.render())


if __name__ == "__main__":
    logging.basicConfig(level=20)

    main_taxi_mcts()
