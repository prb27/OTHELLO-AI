"""
An AI player for Othello. 
"""
import math
import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move
cache = {}


def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)


# Method to compute utility value of terminal state
def compute_utility(board, color):

    util_raw = get_score(board)
    if color == 1:
        return util_raw[0] - util_raw[1]
    return util_raw[1] - util_raw[0]

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    if caching and board in cache:
        return cache[board]

    self_color = 1 if color == 2 else 2

    poss_moves = get_possible_moves(board, self_color)

    if len(poss_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    wrst, curr_wrst_util = (0, 0), math.inf
    for move in poss_moves:
        next_state = play_move(board, self_color, move[0], move[1])
        move_util = minimax_max_node(next_state, color, limit - 1, caching)[1]
        if caching:
            cache[next_state] = move, move_util
        if move_util < curr_wrst_util:
            wrst, curr_wrst_util = move, move_util

    return wrst, curr_wrst_util


def minimax_max_node(board, color, limit, caching = 0):
    if caching and board in cache:
        return cache[board]
    poss_moves = get_possible_moves(board, color)

    if len(poss_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best, curr_best_util = (0, 0), -math.inf
    for move in poss_moves:
        next_state = play_move(board, color, move[0], move[1])
        move_util = minimax_min_node(next_state, color, limit - 1, caching)[1]
        if caching:
            cache[next_state] = move, move_util
        if move_util > curr_best_util:
            best, curr_best_util = move, move_util

    return best, curr_best_util


def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    cache.clear()
    return minimax_max_node(board, color, limit, caching)[0]


def order_moves(board, color, max):
    poss_moves = get_possible_moves(board, color)
    moves = []

    for move in poss_moves:
        moves.append((move, compute_utility(play_move(board, color, move[0], move[1]),  max)))

    if color == max:
        moves.sort(key=lambda x: x[1], reverse=True)
    else:
        moves.sort(key=lambda x: x[1], reverse=False)

    returned = []

    for move in moves:
        returned.append(move[0])

    return returned


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    if caching and board in cache:
        return cache[board]

    self_color = 1 if color == 2 else 2

    if ordering:
        poss_moves = order_moves(board, self_color, color)
    else:
        poss_moves = get_possible_moves(board, self_color)

    if len(poss_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    wrst, curr_wrst_util = (0, 0), math.inf

    for move in poss_moves:
        next_state = play_move(board, self_color, move[0], move[1])
        move_util = alphabeta_max_node(next_state, color, alpha, beta, limit - 1, caching, ordering)[1]
        if caching:
            cache[next_state] = move, move_util
        if move_util < curr_wrst_util:
            wrst, curr_wrst_util = move, move_util

        beta = min(beta, move_util)
        if beta <= alpha:
            break

    return wrst, curr_wrst_util


def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    if caching and board in cache:
        return cache[board]

    if ordering:
        poss_moves = order_moves(board, color, color)
    else:
        poss_moves = get_possible_moves(board, color)

    if len(poss_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best, curr_best_util = (0, 0), -math.inf
    for move in poss_moves:
        next_state = play_move(board, color, move[0], move[1])
        move_util = alphabeta_min_node(next_state, color, alpha, beta, limit - 1, caching, ordering)[1]
        if caching:
            cache[next_state] = move, move_util
        if move_util > curr_best_util:
            best, curr_best_util = move, move_util

        alpha = max(alpha, move_util)
        if beta <= alpha:
            break

    return best, curr_best_util


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    cache.clear()
    return alphabeta_max_node(board, color, -math.inf, math.inf, limit, caching, ordering)[0]


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
