import ai_dm.Search.mcts.mcts_policies as mp
from ai_dm.Search.mcts import MCTS


def mcts_factory(
        set_env_to_state,
        select_and_expand=mp.standard_tree_policy(
            mp.uct_action_selection(
                exploration_constant=1,
                lower_bound_reward=0,
                upper_bound_reward=1)
        ),
        rollout=mp.standard_default_policy(max_depth=100),
        backprop=mp.standard_backprop,
        best_action=mp.standard_best_action_selection,
        prune_tree=None
):

    set_env_to_state_func = set_env_to_state
    select_and_expand_func = select_and_expand
    rollout_func = rollout
    backprop_func = backprop
    best_action_func = best_action
    prune_tree_func = prune_tree

    def factory():

        return MCTS(
            set_env_to_state_func,
            select_and_expand_func,
            rollout_func,
            backprop_func,
            best_action_func,
            prune_tree_func
        )

    return factory
