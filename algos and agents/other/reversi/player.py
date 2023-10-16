class MyPlayer():
    '''dumb_val_mover''' # TODO a short description of your player
    
    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'dumb_val_mover' #TODO: fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size

    def move(self,board):
        # TODO: write you method
        # you can implement auxiliary fucntions, of course
        square_value = [
	    18,  4,  16, 12, 12, 16,  4, 18,
	     4,  2,   6,  8,  8,  6,  2,  4,
	    16,  6,  14, 10, 10, 14,  6, 16,
	    12,  8,  10,  0,  0, 10,  8, 12,
	    12,  8,  10,  0,  0, 10,  8, 12,
	    16,  6,  14, 10, 10, 14,  6, 16,
	     4,  2,   6,  8,  8,  6,  2,  4,
	    18,  4,  16, 12, 12, 16,  4, 18
        ]

        valid = self.get_all_valid_moves(board)
        curr_best_val = 0
        for i in range(len(valid)):
            if square_value[valid[i][0] * 8 + valid[i][1]] > curr_best_val:
                curr_best_val = square_value[valid[i][0] * 8 + valid[i][1]]
                curr_best_move = i
        print(valid)
        return (valid[curr_best_move])

    def __is_correct_move(self, move, board):
        dx = [-1, -1, -1, 0, 1, 1,  1,  0]
        dy = [-1,  0,  1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board)[0]:
                return True, 
        return False

    def __confirm_direction(self, move, dx, dy, board):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == self.opponent_color:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == self.my_color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
    
