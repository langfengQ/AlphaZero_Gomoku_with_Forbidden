import numpy as np
from collections import deque

class BoardBuffer(object):
    '''
    board for the game
    '''
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 11))
        self.height = int(kwargs.get('height', 11))
        self.states = {}
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]
        # player1 and player2

        self.feature_planes = 8
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1,-1]] * self.feature_planes)

        self.color_dict = {'White':1, 'Black':-1, 'Blank':0}
        self.table_2d = np.zeros([self.height+2, self.width+2], dtype=int)

    def init_board(self, start_player=0):
        '''
        init the board and set some variables
        '''
        self.start_player = self.players[start_player]
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)

    def current_state_forbidden(self):
        square_state = self.current_state()
        if square_state[-1].sum() < 0.5:
            table_2d_version = (square_state[1] - square_state[0]).astype(np.int)
        else:
            table_2d_version = (square_state[0] - square_state[1]).astype(np.int)

        return table_2d_version

    def current_state(self):
        '''
        return the board state from the perspective of the current player.
        state shape: (self.feature_planes+1) x width x height
        '''
        square_state = np.zeros((self.feature_planes+1, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            for i in range(self.feature_planes):
                # put all moves on planes
                if i%2 == 0:
                    square_state[i][move_oppo // self.width,move_oppo % self.height] = 1.0
                else:
                    square_state[i][move_curr // self.width,move_curr % self.height] = 1.0
            # delete some moves to construct the planes with history features
            for i in range(0,len(self.states_sequence)-2,2):
                for j in range(i+2,len(self.states_sequence),2):
                    if self.states_sequence[i][1]!= -1:
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] == 1.0, 'wrong oppo number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.
            for i in range(1,len(self.states_sequence)-2,2):
                for j in range(i+2,len(self.states_sequence),2):
                    if self.states_sequence[i][1] != -1:
                        assert square_state[j][self.states_sequence[i][0] // self.width,self.states_sequence[i][0] % self.height] ==1.0, 'wrong player number'
                        square_state[j][self.states_sequence[i][0] // self.width, self.states_sequence[i][0] % self.height] = 0.

        if len(self.states) % 2 == 0:
            square_state[self.feature_planes][:, :] = 1.0  # indicate the colour to play 1.0-balck, 0.0-white
        return square_state[:, ::-1, :]

    def do_move(self, move):
        '''
        update the board
        '''
        self.states[move] = self.current_player

        self.states_sequence.appendleft([move,self.current_player])

        self.availables.remove(move)

        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        '''
        judge if there's a 5-in-a-row, and which player if so
        '''
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            # too few moves to get 5-in-a-row
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player
                
        # check forbidden
        table_2d = self.current_state_forbidden()
        self.table_2d[1:16,1:16] = table_2d

        act = list(states.keys())[-1]

        player = states[act]
        start_player = states[list(states.keys())[0]]
        assert self.start_player == start_player, 'start player is wrong'
        if player == start_player: ## black player == 1, white player == 2
            pos_x, pos_y = self.transfer_move(act)
            # edge adjust
            pos_x, pos_y = pos_x+1, pos_y+1
            pos = [pos_y, pos_x]
            if not self.check_forbidden(pos, "Black"):
                white_player = self.players[0] if start_player == self.players[1] else self.players[1]
                return True, white_player

        return False, -1

    def check_forbidden(self, pos, color):
        if self.longlink_info(pos, color) == False:
            return False
        elif self.huosi_info(pos, color) == False:
            return False
        elif self.huosan_info(pos, color) == False:
            return False
        else:
            return True

    def game_end(self):
        '''
        Check whether the game is end
        '''
        end, winner = self.has_a_winner()
        if end:
            # if one win,return the winner
            return True, winner
        elif not len(self.availables):
            # if the board has been filled and no one win ,then return -1
            return True, -1
        return False, -1

    def get_current_player(self):
        '''
        return current player
        '''
        return self.current_player
        
    def transfer_move(self, move):
        y = move % self.width
        x = 14 - (move // self.height)

        return x, y

    def huosan_info(self, pos, color):
        '''
        huo_san
        --xxx--
        --x-xx-
        --xx-x-
        '''
        color_num = self.color_dict[color]
        color_bnk = self.color_dict['Blank']
        huosan_row = []
        huosi_col = []
        huosi_dig = []
        huosi_adig = []
        huosan_all = []

        # check row
        for i in range(6):
            if  (not self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+4) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+5) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+6) == color_bnk ) \
                or \
                (    self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+6) == color_num ) \
                or \
                (not self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+6) == color_num ) :
                    chess = [[pos[1], pos[0]-5+i+x] for x in range(7) if self.outbound_value(self.table_2d, pos[1], pos[0]-5+i+x) == color_num]
                    if chess not in huosan_row: 
                        huosan_row.append(chess)
                        huosan_all.append(chess)
                        #print('huosan_row', huosan_row)

        # check col
        for i in range(6):
            if  (not self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]) == color_bnk ) \
                or \
                (    self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]) == color_num ) \
                or \
                (not self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]) == color_num ) :
                    chess = [[pos[1]-5+i+x, pos[0]] for x in range(7) if self.outbound_value(self.table_2d, pos[1]-5+i+x, pos[0]) == color_num]
                    if chess not in huosi_col:   
                        huosi_col.append(chess)
                        huosan_all.append(chess)
                        #print('huosi_col', huosi_col)

        # check dig
        for i in range(6):
            if  (not self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]-5+i+4) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]-5+i+5) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]-5+i+6) == color_bnk ) \
                or \
                (    self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]-5+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]-5+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]-5+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]-5+i+6) == color_num ) \
                or \
                (not self.outbound_value(self.table_2d, pos[1]-5+i+0, pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+1, pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+2, pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+4, pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-5+i+5, pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-5+i+6, pos[0]-5+i+6) == color_num ) :
                    chess = [[pos[1]-5+i+x, pos[0]-5+i+x] for x in range(7) if self.outbound_value(self.table_2d, pos[1]-5+i+x, pos[0]-5+i+x) == color_num]
                    if chess not in huosi_dig:   
                        huosi_dig.append(chess)
                        huosan_all.append(chess)
                        #print('huosi_dig', huosi_dig)

        # check anti-dig
        for i in range(6):
            if  (not self.outbound_value(self.table_2d, pos[1]+5-i-0, pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-1, pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-2, pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-4, pos[0]-5+i+4) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-5, pos[0]-5+i+5) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-6, pos[0]-5+i+6) == color_bnk ) \
                or \
                (    self.outbound_value(self.table_2d, pos[1]+5-i-0, pos[0]-5+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-1, pos[0]-5+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-2, pos[0]-5+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-4, pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-5, pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]+5-i-6, pos[0]-5+i+6) == color_num ) \
                or \
                (not self.outbound_value(self.table_2d, pos[1]+5-i-0, pos[0]-5+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-1, pos[0]-5+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-2, pos[0]-5+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-3, pos[0]-5+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-4, pos[0]-5+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+5-i-5, pos[0]-5+i+5) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]+5-i-6, pos[0]-5+i+6) == color_num ) :
                    chess = [[pos[1]+5-i-x, pos[0]-5+i+x] for x in range(7) if self.outbound_value(self.table_2d, pos[1]+5-i-x, pos[0]-5+i+x) == color_num]
                    if chess not in huosi_adig:  
                        huosi_adig.append(chess)
                        huosan_all.append(chess)
                        #print('huosi_adig', huosi_adig)
        #print(huosan_all)
        if len(huosan_all) >= 2:
            self.forbidden_flag = 'sansan'
            return False

    def huosi_info(self, pos, color):
        '''
        chong_si
        oxxxx-
        -xxxxo
        oxxx-x
        x-xxxo
        oxx-xx
        xx-xxo
        ox-xxx
        xxx-xo 
        huo_si
        -xxxx-

        class1
        xxxx-? 
        class2
        -xxxx?
        class3
        xxx-x?
        class4
        xx-xx?
        class5
        x-xxx?
        '''
        color_num = self.color_dict[color]
        color_bnk = self.color_dict['Blank']
        huosi_row = []
        huosi_col = []
        huosi_dig = []
        huosi_adig = []
        huosi_all = []
 
        # check row
        for i in range(5):
            if  (    self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == color_num) :
                    chess = [[pos[1], pos[0]-4+i+x] for x in range(6) if self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+x) == color_num]
                    if chess not in huosi_row:
                        huosi_row.append(chess)
                        huosi_all.append(chess)
                        #print('huosi_row', huosi_row)

        # check col
        for i in range(5):
            if  (    self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == color_num) :
                    chess = [[pos[1]-4+i+x, pos[0]] for x in range(6) if self.outbound_value(self.table_2d, pos[1]-4+i+x, pos[0]) == color_num]
                    if chess not in huosi_col:
                        huosi_col.append(chess)
                        huosi_all.append(chess)
                        #print('huosi_col', huosi_col)

        # check dig
        for i in range(5):
            if  (    self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == color_num) :
                    chess = [[pos[1]-4+i+x, pos[0]-4+i+x] for x in range(6) if self.outbound_value(self.table_2d, pos[1]-4+i+x, pos[0]-4+i+x) == color_num]
                    if chess not in huosi_dig:
                        huosi_dig.append(chess)
                        huosi_all.append(chess)
                        #print('huosi_dig', huosi_dig)

        # check anti-dig
        for i in range(5):
            if  (    self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) == color_bnk and \
                 not self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == color_num) \
            or \
                (    self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) == color_num and \
                 not self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == color_num) \
            or \
                (not self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) == color_bnk and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) == color_num and \
                     self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == color_num) :
                    chess = [[pos[1]+4-i-x, pos[0]-4+i+x] for x in range(6) if self.outbound_value(self.table_2d, pos[1]+4-i-x, pos[0]-4+i+x) == color_num]
                    if chess not in huosi_adig:    
                        huosi_adig.append(chess)
                        huosi_all.append(chess)
                        #print('huosi_adig', huosi_adig)

        #print(huosi_all)
        if len(huosi_all) >= 2:
            self.forbidden_flag = 'sisi'
            return False

    def longlink_info(self, pos, color):
        '''
        six link
        xxxxxx
        '''
        color_num = self.color_dict[color]
 
        # check row
        for i in range(5):
            if  self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+0) + \
                self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+1) + \
                self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+2) + \
                self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+3) + \
                self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+4) + \
                self.outbound_value(self.table_2d, pos[1], pos[0]-4+i+5) == 6*color_num:
                    self.forbidden_flag = 'longlink'
                    return False

        # check col
        for i in range(5):
            if  self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]) == 6*color_num:
                    self.forbidden_flag = 'longlink'
                    return False

        # check dig
        for i in range(5):
            if  self.outbound_value(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4) + \
                self.outbound_value(self.table_2d, pos[1]-4+i+5, pos[0]-4+i+5) == 6*color_num:
                    self.forbidden_flag = 'longlink'
                    return False

        # check anti-dig
        for i in range(5):
            if  self.outbound_value(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0) + \
                self.outbound_value(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1) + \
                self.outbound_value(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2) + \
                self.outbound_value(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3) + \
                self.outbound_value(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4) + \
                self.outbound_value(self.table_2d, pos[1]+4-i-5, pos[0]-4+i+5) == 6*color_num:
                    self.forbidden_flag = 'longlink'
                    return False

    def check_win(self, pos, color):
        color_num = self.color_dict[color]
        #pos[1] table de hang, gc de lie
        #pos[0] table de lie, gc de hang

        # check row '---'
        for i in range(5):
            if  self.outbound_equal(self.table_2d, pos[1], pos[0]-4+i+0, color_num) and \
                self.outbound_equal(self.table_2d, pos[1], pos[0]-4+i+1, color_num) and \
                self.outbound_equal(self.table_2d, pos[1], pos[0]-4+i+2, color_num) and \
                self.outbound_equal(self.table_2d, pos[1], pos[0]-4+i+3, color_num) and \
                self.outbound_equal(self.table_2d, pos[1], pos[0]-4+i+4, color_num) :
                    return True

        # check col '|'
        for i in range(5):
            if  self.outbound_equal(self.table_2d, pos[1]-4+i+0, pos[0], color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+1, pos[0], color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+2, pos[0], color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+3, pos[0], color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+4, pos[0], color_num) :
                    return True

        # check diagonal '\' 
        for i in range(5):
            if  self.outbound_equal(self.table_2d, pos[1]-4+i+0, pos[0]-4+i+0, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+1, pos[0]-4+i+1, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+2, pos[0]-4+i+2, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+3, pos[0]-4+i+3, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]-4+i+4, pos[0]-4+i+4, color_num) :
                    return True

        # check anti-diagonal '/'
        for i in range(5):
            if  self.outbound_equal(self.table_2d, pos[1]+4-i-0, pos[0]-4+i+0, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]+4-i-1, pos[0]-4+i+1, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]+4-i-2, pos[0]-4+i+2, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]+4-i-3, pos[0]-4+i+3, color_num) and \
                self.outbound_equal(self.table_2d, pos[1]+4-i-4, pos[0]-4+i+4, color_num) :
                    return True

        return False

    def outbound_value(self, table, x, y):
        # self.width, self.height
        if x == 0 or x == (self.height + 1) or \
           y == 0 or y == (self.width + 1) :
            return 999
        else:
            try:
                return table[x][y]
            except:
                return 999

    def outbound_equal(self, table, x, y, pt):
        try:
            return (table[x][y] == pt)
        except:
            return False