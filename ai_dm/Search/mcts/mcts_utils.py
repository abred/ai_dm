import ai_dm.Search.mcts.mcts_policies as mp
from ai_dm.Search.mcts import MCTS


def mcts_factory(
        budget,
        update_budget=lambda x: x-1,
        select_and_expand=mp.standard_tree_policy(
            mp.uct_action_selection,
        ),
        rollout=mp.standard_default_policy,
        backprop=mp.standard_backprop,
        best_action=mp.standard_best_action_selection):

    budget_val = budget
    update_budget_func = update_budget
    select_and_expand_func = select_and_expand
    rollout_func = rollout
    backprop_func = backprop
    best_action_func = best_action

    def factory():

        return MCTS(
            budget_val,
            update_budget_func,
            select_and_expand_func,
            rollout_func,
            backprop_func,
            best_action_func
        )

    return factory
