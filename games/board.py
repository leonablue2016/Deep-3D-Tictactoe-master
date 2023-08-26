import numpy as np

def empty_state(dims):
    '''establish the empty state wherein each cell is filled by zero'''
    #return np.array([[0] * 3] * 3)
    seed = 0
    for i in range(dims-1):
        seed = [[seed] * 4] * 4
    return np.array(seed)

def print_board(state):
    if len(state) == 3:
        board_format = "-------------\n| {0} | {1} | {2} |\n|-------------|\n| {3} | {4} | {5} |\n|-------------|\n| {6} | {7} | {8} |\n-------------"
    else:
        board_format = "----------------\n| {0} | {1} | {2} | {3} |\n|---------------|\n| {4} | {5} | {6} | {7} |\n|---------------|\n| {8} | {9} | {10} | {11} |\n----------------|\n| {12} | {13} | {14} | {15} |\n----------------"   
    cell_values = []
    symbols = ["O", " ", "X"]

    for i in range(len(state)):
        for j in range(len(state[i])):
            cell_values.append(symbols[int(state[i][j] + 1)])

    print(board_format.format(*cell_values))

def flatten_state(state):
    return np.transpose(np.vstack([cell for cell in state.flat]))

def open_spots(state):
   
    open_cells = []
    flat_state = [cell for cell in state.flat]
    
    for i in range(len(flat_state)):
        if flat_state[i] == 0:
            open_cells.append(i)
    
    '''if len(open_cells) < 78:
        print(flat_state)'''


    return open_cells

def is_game_over(state):

    if len(state.shape) > 2:
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                winner = game_status(state)
                
                if winner != 0:
                    return winner

        return winner
    
    else:
        return game_status(state)

def game_status(state):
    ''' check if any of the columns or rows or diagonals when summed are
    divisible by the width or height of the board - indication that the
    game has been won. The sign of the sum tells us who has won'''

    for i in range(len(state)):

        state_trans = np.array(state).transpose()  # transposed board state

        # check for winner row-wise
        if np.array(state[i]).sum() != 0 and np.array(state[i]).sum() % 4 == 0:
            return np.array(state[i]).sum() / 4

            # check for winner column-wise
        elif state_trans[i].sum() != 0 and state_trans[i].sum() % 4 == 0:
            return state_trans.sum() / 4

    # extract major diagonal from the state
    major_diag = np.multiply(np.array(state), np.identity(len(state)))
    if major_diag.sum() != 0 and major_diag.sum() % 4 == 0:
        return major_diag.sum() / 4

    # extract minor diagonal from the state
    minor_diag = np.multiply(np.array(state), np.fliplr(major_diag))
    if minor_diag.sum() != 0 and minor_diag.sum() % 4 == 0:
        return minor_diag.sum() / 4

    return 0  # no clear winner


def get_state_key(state):
    ''' convert the state into a unique key by concatenating
    all the flattened values in the state'''

    flat_state = [cell for cell in state.flat]
    key = "".join(map(str, flat_state))
    #print(key)
    return key
