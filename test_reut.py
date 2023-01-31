import util
from connect4 import GameState


def test():
    state = GameState()
    board = [[1, 1, 2, 2, 0, 1, 0],[0,1,2,2,0,2,0],[0,0,1,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    state.set_board_AIturn(board)
