import logging

import gym
# from ai_dm.Environments.gym_envs.gym_problem import GymProblem

import ai_dm.Search.mcts.mcts_policies as mp
from ai_dm.Search.mcts import mcts_factory

logger = logging.getLogger(__name__)


def main_taxi_mcts():

    # define the environment
    env = gym.make("Taxi-v3", render_mode='ansi').env
    init_state = env.reset()[0]

    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(init_state)
    print(taxi_row, taxi_col, pass_idx, dest_idx)

    print("initial state:", init_state)
    print(env.render())

    budget = 500
    max_depth = 100
    exploration_constant = 0.5
    lower_bound_reward = -10 * max_depth
    upper_bound_reward = 20

    update_budget = lambda x: x-1
    select_and_expand = mp.standard_tree_policy(
        mp.uct_action_selection(
                exploration_constant=exploration_constant,
                lower_bound_reward=lower_bound_reward,
                upper_bound_reward=upper_bound_reward)
    )
    rollout = mp.standard_default_policy(max_depth=max_depth)
    backprop = mp.standard_backprop
    best_action = mp.standard_best_action_selection

    # perform MCTS
    mcts = mcts_factory(
        budget,
        update_budget,
        select_and_expand,
        rollout,
        backprop,
        best_action
    )()

    state = init_state
    terminated = False
    while not terminated:
        action = mcts.get_action(state, env)
        logger.info("Applying action: %s", action)
        if action is None:
            env.render()
            break
        state, _, terminated, _, _ = env.step(action)

        print(env.render())


if __name__ == "__main__":
    logging.basicConfig(level=20)

    main_taxi_mcts()
