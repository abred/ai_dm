__author__ = 'sarah'


# adapdted from aima-python-master: https://github.com/aimacode/aima-python

import collections, queue
import bisect

from abc import ABC, abstractmethod


class State:

    """State superclass
    """
    def __init__(self, key, is_terminal=False):
        self.key = key
        self.is_terminal = is_terminal

    def get_key(self):
        return self.key

    def __str__(self):
        return self.key

    def __repr__(self):
        return self.key

    def is_terminal(self):
        return self.is_terminal


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual content for this node.
    Also specifies the transtion that got us to this state, and the total path_cost (also known as g) to reach the node.
    Other functions may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent, action, path_cost, info=None):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        self.info = info

    def __repr__(self):
        return "<Node {}>".format(self.state.__str__())

    def __lt__(self, node):
        return self.state.get_key() < node.state.get_key()

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def get_transition_path(self):
        """Return a list of transitions forming the execution path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node.action)
            node = node.parent
        return list(reversed(path_back))

    def get_transition_path_string(self):
        """Return a list of transitions forming the execution path from the root to this node."""
        node, path_back = self, []
        while node:
            action_name = 'None'
            if node.action:
                action_name = node.action.__str__()
            if action_name != 'None':
                path_back.append(action_name)

            node = node.parent
        return list(reversed(path_back))

    def get_path_cost(self, problem):
        """Return the total cost of the list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        cost = 0
        while node:
            path_back.append(node)
            if node.action is not None:
                cost = cost + problem.get_action_cost(node.action, node.state)
            node = node.parent
        return [cost, list(reversed(path_back))]

    def get_path_value(self, problem):
        """Return the total value pf the list of states forming the path from the root to this node."""
        node, path_back = self, []
        value = 0
        while node:
            path_back.append(node)
            if node.action is not None:
                value = value + problem.get_action_value(node.action, node.state)
            node = node.parent
        return [value, list(reversed(path_back))]

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class Container(ABC):

    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
     """
    def __init__(self, container, max_len=None):

        self.container = container
        self.max_len = max_len


    @abstractmethod
    def add(self, item, check_existance=False):
        pass

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __contains__(self, item):
        pass

    @abstractmethod
    def __repr__(self, key):
        pass

    def is_empty(self):
        True if self.__len__() == 0 else False


'''A First-In-First-Out Queue'''
class FIFOQueue(Container):

    def __init__(self, max_len=None):
        super().__init__(queue.Queue(), max_len)

    def add(self, item, check_existance=False):

        if check_existance and self.__contains__(item):
            return

        if not self.max_len or self.__len__() < self.max_len:
            self.container.put(item)
        else:
            raise Exception('FIFOQueue is full')

    def extract(self):
        return self.container.get()


    def __len__(self):
        return self.container.qsize

    #todo: check if this works
    def __contains__(self, item):
        return item in self.container.queue

    def __repr__(self):
        queue_string = ''
        for item in self.container:
            queue_string+= ' '
            queue_string+= item
            
        return queue_string

class PriorityQueue(Container):

    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order=min, f=lambda x: x):
        self.queue = []
        self.order = order
        self.f = f   
    
    #def add_(self, item):
    #    bisect.insort(self.A, (self.f(item), item))
    
    def add(self, node):        
        node.heuristic_value = self.f(node)
        bisect.insort(self.queue, (node.heuristic_value, node))
        
    def __len__(self):
        return len(self.queue)

    def extract(self):
        #print('queue is ')
        #print(self.__repr__())
  
        if self.order == min:
            
            return self.queue.pop(0)[1]
        else:
            return self.queue.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.queue)

    def __getitem__(self, key):
        for _, item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.queue):
            if item == key:
                self.queue.pop(i)
                
    def __repr__(self):
        queue_string = ''
        for item  in self.queue:          
        
            queue_string+= '( %d - %s)'%(item[0],item[1])
            
        return queue_string
            

class ClosedList(ABC):

    # Holding the list of items that have been explored
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add(self, node):
        pass

    @abstractmethod
    def isInList(self, node):
        pass


class ClosedListOfSequences(ClosedList):
    ''' Holding the list of items that have been explored 
    '''
    def __init__(self):
        self.closed_list = []

    def add(self, node):
        self.closed_list.append(node.state)

    def isInList(self, node):
        
        if node.state not in self.closed_list:
            return False
        else:
            return True


class ClosedListOfSets(ClosedList):
    ''' Holding the list of items that have been explored 
    '''
    def __init__(self):
        self.closed_list = []

    def add(self, node):
        sequence = node.transition_path()
        sorted_sequence = sorted(sequence)
        self.closed_list.append(sorted_sequence)

    def isInList(self, node):
        sequence = node.transition_path()
        sorted_sequence = sorted(sequence)
        if sorted_sequence not in self.closed_list:
            return False
        else:
            return True       


class TerminationCriteria(ABC):

    @abstractmethod
    def isTerminal(self, node, value,problem):
        pass



class TerminationCriteriaOptimalValue(TerminationCriteria):

    def __init__(self, optimal_value, orSmaller=True):
        self.optimal_value = optimal_value
        self.orSmaller = orSmaller

    def isTerminal(self, node, value,problem):
        if self.orSmaller:
            if node.value <= self.optimal_value:
                return True
            else:
                return False

        else:  # or bigger
            if node.value >= self.optimal_value:
                return True
            else:
                return False

    def __str__(self):
        raise NotImplementedError


class TerminationCriteriaGoalStateReached(TerminationCriteria):

    def isTerminal(self, node, value,problem):
        if problem.is_goal_state(node.state):
            return True
        else:
            return False


class EvaluationCriteria(ABC):

    @abstractmethod
    def is_better_or_equal(self, value_a, node_a, value_b, node_b, problem):
        pass


class EvaluationCriteriaGoalCondition(EvaluationCriteria):

    def is_better_or_equal(self, value_a, node_a, value_b, node_b, problem):

        if node_a.state.is_terminal:
            if not node_b or not node_b.state.is_terminal or node_b.state.is_terminal and problem.is_better_or_equal(value_a, value_b) :
                return True

        return False

    def __str__(self):
        raise NotImplementedError
