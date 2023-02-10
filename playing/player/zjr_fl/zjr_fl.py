import time
from player.zjr_fl.mcts_reinforcement import MCTSPlayer
from player.zjr_fl.policy_value_resnet import PolicyValueNet
from player.zjr_fl.board_buffer import BoardBuffer
import numpy as np

class ZJR_FL():

    def __init__(self):
        
        self.playerjudger = None
        self.table_2d = None
        self.color = None
        self.anticolor = None
        
        self.color_dict = {'White':1, 'Black':-1, 'Blank':0}
        self.t1 = None
        
        n = 5
        self.width, self.height = 15, 15
        # ResNet model path
        model_file = './player/zjr_fl/policy_value_model/best_policy.model'

        policy_value_net = PolicyValueNet(board_width=self.width,board_height=self.height,block=19,init_model=model_file,cuda=False)

        self.mctsplayer = MCTSPlayer(policy_value_function=policy_value_net.policy_value_function,
                                    action_fc=policy_value_net.action_fc_test,
                                    evaluation_fc=policy_value_net.evaluation_fc2_test,
                                    c_puct=5, 
                                    n_playout=1200, # you can set any positive number (the larger -> the stronger)
                                    )

        self.board_buffer = BoardBuffer(width=self.width, height=self.height, n_in_row=n)

        self.mctsplayer.reset_player()
        self.board_buffer.init_board()

    def xiazi(self, playerjudger, color, step):
        print()
        print('-' * 30, "  zjr_fl's turn begins ! ", '-' * 30)
        if color == 'Black':
            self.color = 'Black'
            self.anticolor = 'White'
        else:
            self.color = 'White'
            self.anticolor = 'Black'

        # first step of black
        if step == 0:
            pos_x = 7
            pos_y = 7
        else:
            self.playerjudger = playerjudger
            moves = self.get_moves(playerjudger.table_2d[1:16,1:16], self.table_2d, color)
            if len(moves) != 0:
                for m in moves:
                    self.board_buffer.do_move(m)
            self.table_2d = playerjudger.table_2d[1:16,1:16]
            # 黑-1，白1

            self.t1 = time.perf_counter()
            move = self.mctsplayer.get_action(self.board_buffer, print_info=True)
            pos_x, pos_y = self.board_buffer.transfer_move(move)
        # edge adjust
        pos_x, pos_y = pos_x+1, pos_y+1
        print('-' * 31, "  zjr_fl's turn ends ! ", '-' * 31)
        print()
        return pos_x, pos_y

    def get_moves(self, curr_state, old_sate, c):
        color = 1 if c == 'White' else -1
        if old_sate is None:
            state_transition = curr_state.copy()
        else:
            state_transition = curr_state.copy() - old_sate.copy()

        moves_num = len(np.where(state_transition != 0)[0])
        moves = []
        if moves_num == 1:
            x, y = np.where(state_transition == -color)[0], np.where(state_transition == -color)[1]
            pos = (self.height - x.item() - 1) * self.width + y.item()
            moves.append(pos)
        elif moves_num == 2:
            for i in range(moves_num):
                if i == 0:
                    x, y = np.where(state_transition == color)[0], np.where(state_transition == color)[1]
                    pos = (self.height - x.item() - 1) * self.width + y.item()
                    moves.append(pos)
                else:
                    x, y = np.where(state_transition == -color)[0], np.where(state_transition == -color)[1]
                    pos = (self.height - x.item() - 1) * self.width + y.item()
                    moves.append(pos)
        elif moves_num == 0:
            pass
        else:
            assert False, 'get_moves_wrong'
        return moves
