from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import random
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import matplotlib.pyplot as plt
from q_learning import print_board

ticTacToeShape = (3,3,3)
# DataSetInput = np.array((,3,3,3))
PlayerX = "x"
PlayerO = "o"

def generateEmptyBoard():
    return np.zeros(ticTacToeShape).astype("int")


def printBoard3D(board):
    
    for sub_board in board:
        print_board(sub_board)
        print()

def checkIfGameOver(state):
    """
    check if a game is over.
    This function will go to each element, and check if its an 'x' or an 'o'

    :param state: a valid state with strictly 3 dimensions, but can have any shape
    :return: true if game is over , false if game is not over
    """
    for i in xrange(ticTacToeShape[0]):
        for j in xrange(ticTacToeShape[1]):
            for k in xrange(ticTacToeShape[2]):
                if state[i][j][k] != 'x' and state[i][j][k] != 'o':
                    return False
    return True

def hasWonGame(state):
    ''' check if either opponent has won the game given the current board'''
    
    for sub_board in state:
        sub_board = np.array(deepcopy(sub_board))
        '''winner = is_game_over(sub_board)
        if winner != 0:
            return True'''

def getOpenSpots(state):
    """
    gets the list of all open spots for a given state.
    Each element in the list will be a tuple , with ith,jth,kth positions
    :param state:
    :return:
    """
    openSpots = []

    for i in xrange(ticTacToeShape[0]):
        for j in xrange(ticTacToeShape[1]):
            for k in xrange(ticTacToeShape[2]):
                if state[i][j][k] != 'x' and state[i][j][k] != 'o':
                    openSpots.append((i,j,k))

    return openSpots

def makeRandomMove(state):
   """
   get a random tuple from the list of tuples
   :param state:
   :return:
   """
   return random.choice(getOpenSpots(state))


<<<<<<< HEAD
=======

>>>>>>> bb601ab2da342d057377a919141091a78a1fb284
def StartRandomPlay():
    startingPlayer = PlayerX if random.randint(0,1) ==1 else PlayerO



model = Sequential()
model.add(Dense(256,input_dim=ticTacToeShape[0]*ticTacToeShape[1]*ticTacToeShape[2]))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

empty_board = generateEmptyBoard()
printBoard3D(empty_board)
