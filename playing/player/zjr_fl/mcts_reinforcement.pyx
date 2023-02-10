import numpy as np
import copy
import time
from libc.math cimport log as clog
from libc.math cimport sqrt as csqrt
cimport cython

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    '''
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self.c_base = 19652
        self._P = prior_p # its the prior probability that action's taken to get this node

    def expand(self, action_priors):
        '''
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        '''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, float(prob))

    def select(self, int c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    @cython.cdivision(True)
    def update(self, float leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        
    def update_recursive(self, float leaf_value):
        '''
        Like a call to update(), but applied recursively for all ancestors.
        '''
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # every step for revursive update,
        self.update(leaf_value)

    @cython.cdivision(True)
    def get_value(self, int c_puct):

        if self._parent._n_visits < 800:
            C_s = <double>(c_puct)
        else:
            C_s = clog(<double>(1 + self._parent._n_visits + self.c_base) / self.c_base) + <double>(c_puct)
            
        self._u = (C_s * self._P *
                   csqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    '''
    Monte Carlo Tree Search.
    '''
    def __init__(self, policy_value_fn,action_fc,evaluation_fc, is_selfplay,c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(None, 1.0)

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        self._n_playout = n_playout # times of tree search
        self._is_selfplay = is_selfplay

    def _playout(self, state):
        '''
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        '''
        node = self._root
        cdef int deep = 0
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)
            deep += 1

        action_probs, leaf_value = self._policy_value_fn(state,self._action_fc,self._evaluation_fc)
        leaf_value = leaf_value.item()
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:  
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

        return deep

    def get_move_visits(self, state):
        '''
        Run all playouts sequentially and return the available actions and
        their corresponding visiting times.
        state: the current game state
        '''
        t_start = time.time()
        max_search_depth = 0
        show_interval = self._n_playout // 17 if self._n_playout > 20 else 2
        
        cdef int n
        for n in range(self._n_playout+1):
            state_copy = copy.deepcopy(state)
            depth = self._playout(state_copy)
            max_search_depth = depth if depth > max_search_depth else max_search_depth
            if n % show_interval == 0:
                consumed_time = time.time()-t_start
                print ' searching {:.1f}% ...     consumed time: {:.1f}s \r'.format(100*<double>(n)/self._n_playout, consumed_time),

        print ' ' * 80
        # calc the visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        return acts, visits, max_search_depth

    def update_with_move(self, last_move):
        '''
        Step forward in the tree, keeping everything we already know
        about the subtree.
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''
    def __init__(self, policy_value_function,action_fc,evaluation_fc,c_puct=5, n_playout=1600, is_selfplay=False):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay
        self.policy_value_function = policy_value_function
        self.action_fc = action_fc
        self.evaluation_fc = evaluation_fc

        self.mcts = MCTS(policy_value_fn = policy_value_function,
                         action_fc = action_fc,
                         evaluation_fc = evaluation_fc,
                         is_selfplay = self._is_selfplay,
                         c_puct = c_puct,
                         n_playout = n_playout,
                         )

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self,board, print_info):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            self.mcts.update_with_move(board.last_move)

            t = time.time()
            acts, visits, max_search_depth = self.mcts.get_move_visits(board)
            search_time = time.time() - t

            temp = 1e-3
            # always choose the most visited move
            probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
            move = np.random.choice(acts, p=probs)

            self.mcts.update_with_move(move)
            eval_win = self.mcts._root._Q / 2 + 0.5
            # update the tree with self move

            if print_info:
                print ' ' * 10 + 'step:                             {}'.format(225 - len(sensible_moves))
                my_x, my_y = board.transfer_move(move)
                print ' ' * 10 + "zjr_fl's decision:                x = {} y = {}".format(my_x+1, my_y+1)
                print ' ' * 10 + 'decision simulations:             {}'.format(visits[np.where(acts == move)[0].item()])
                print ' ' * 10 + 'total simulations:                {}'.format(np.sum(np.array(visits)))
                print ' ' * 10 + 'max search depth:                 {}'.format(max_search_depth)
                if len(sensible_moves) <= 215:
                    print ' ' * 10 + "zjr_fl's eval winning rate:       {:.2f}%".format(100 * eval_win)
                else:
                    print ' ' * 10 + "zjr_fl's eval winning rate:       None (will be available after 10 steps)"
                print ' ' * 10 + "search time:                      {:.1f}s".format(search_time)
                print 


            return move

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha {}".format(self.player)


