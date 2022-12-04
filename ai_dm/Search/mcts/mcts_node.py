import ai_dm.Search.utils as utils


class MCTSNode(utils.Node):
    def __init__(
            self,
            state,
            terminal,
            parent=None,
            action=None,
            reward=0,
    ):
        super().__init__(state, parent, action)

        self.terminal = terminal
        self.reward = reward
        if self.parent is not None:
            self.cum_reward = reward + self.parent.cum_reward
        else:
            self.cum_reward = reward
        self.children = {}
        self.seen_values = []
        self.count_visited = 0
        self.count_action_tried = {}
        self.value_action_tried = {}

    def __repr__(self):
        s = (
            "MCTSNode: "
            f"State {self.state}, "
            f"Terminal? {self.terminal}, "
            f"Parent {self.parent.state if self.parent is not None else None}, "
            f"Action {self.action}, "
            f"Reward {self.reward}, "
            f"Visited {self.count_visited} times, "
            f"Cum. Reward {self.cum_reward}, "
            f"Children {[child.state for child in self.children.values()]}, "
            f"Tries per action {self.count_action_tried}, "
            f"Value per action {self.value_action_tried}, "
            f"Values seen so far {self.seen_values}."
            )
        return s
