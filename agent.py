"""
An AI player for Othello. 
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

minimax_val = {}

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)


# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    dark, light = get_score(board)
    if color == 1:  # player is black
        return dark - light
    else:  # color == 2 (i.e., player is white)
        return light - dark


# Better heuristic value of board
def compute_heuristic(board, color): #not implemented, optional
    #IMPLEMENT
    return 0 #change this!


############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching=0):
    #IMPLEMENT

    if caching == 1 and board in minimax_val:
        return minimax_val[board]

    next_color = 2 if color == 1 else 1

    moves = get_possible_moves(board, next_color)
    terminal = (len(moves) == 0)

    if terminal or limit == 0:
        return None, compute_utility(board, color)
    else:
        utilities = []
        for move in moves:
            successor_state = play_move(board, next_color, move[0], move[1])
            u = minimax_max_node(successor_state, color, limit - 1)[1]
            utilities.append(u)

        best_move = moves[utilities.index(min(utilities))], min(utilities)
        if caching == 1:
            minimax_val[board] = best_move  # cache minimax value

        return best_move


def minimax_max_node(board, color, limit, caching=0): #returns highest possible utility
    #IMPLEMENT

    if caching == 1 and board in minimax_val:
        return minimax_val[board]

    moves = get_possible_moves(board, color)
    terminal = (len(moves) == 0)

    if terminal or limit == 0:
        return None, compute_utility(board, color)
    else:
        utilities = []
        for move in moves:
            successor_state = play_move(board, color, move[0], move[1])
            u = minimax_min_node(successor_state, color, limit-1)[1]
            utilities.append(u)

        best_move = moves[utilities.index(max(utilities))], max(utilities)
        if caching == 1:
            minimax_val[board] = best_move  # cache minimax value

        return best_move


def select_move_minimax(board, color, limit=sys.maxsize, caching=0):
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
    #IMPLEMENT

    # ignore limit and caching for now; select move based on plain minimax strategy
    # Implement recursively using minimax_max_node(board, color) and minimax_min_node(board, color)

    moves = get_possible_moves(board, color)

    utilities = []
    for move in moves:
        successor_state = play_move(board, color, move[0], move[1])
        u = minimax_min_node(successor_state, color, limit, caching)[1]
        utilities.append(u)

    optimal_action = moves[utilities.index(max(utilities))]

    return optimal_action


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    #IMPLEMENT

    if caching == 1 and board in minimax_val:
        return minimax_val[board]

    next_color = 2 if color == 1 else 1

    moves = get_possible_moves(board, next_color)
    terminal = (len(moves) == 0)

    if terminal or limit == 0:
        return None, compute_utility(board, color)
    else:
        if ordering == 1:
            ordered_moves = []
            for move in moves:
                successor_state = play_move(board, next_color, move[0], move[1])
                ordered_moves.append((compute_utility(successor_state, color), move))
            ordered_moves.sort(reverse=True)
            moves = []
            for utility_move in ordered_moves:
                moves.append(utility_move[1])

        utilities = []
        for move in moves:
            successor_state = play_move(board, next_color, move[0], move[1])
            u = alphabeta_max_node(successor_state, color, alpha, beta, limit - 1, caching, ordering)[1]
            beta = min(beta, u)
            utilities.append(u)
            if beta <= alpha:
                break

        best_move = moves[utilities.index(min(utilities))], beta
        if caching == 1:
            minimax_val[board] = best_move  # cache minimax value

        return best_move


def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    #IMPLEMENT

    if caching == 1 and board in minimax_val:
        return minimax_val[board]

    moves = get_possible_moves(board, color)
    terminal = (len(moves) == 0)

    if terminal or limit == 0:
        return None, compute_utility(board, color)
    else:
        if ordering == 1:
            ordered_moves = []
            for move in moves:
                successor_state = play_move(board, color, move[0], move[1])
                ordered_moves.append((compute_utility(successor_state, color), move))
            ordered_moves.sort(reverse=True)
            moves = []
            for utility_move in ordered_moves:
                moves.append(utility_move[1])

        utilities = []
        for move in moves:
            successor_state = play_move(board, color, move[0], move[1])
            u = alphabeta_min_node(successor_state, color, alpha, beta, limit-1, caching, ordering)[1]
            alpha = max(alpha, u)
            utilities.append(u)
            if beta <= alpha:
                break

        best_move = moves[utilities.index(max(utilities))], alpha
        if caching == 1:
            minimax_val[board] = best_move  # cache minimax value

        return best_move


def select_move_alphabeta(board, color, limit=sys.maxsize, caching=0, ordering=0):
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
    #IMPLEMENT
    alpha = float("-inf")
    beta = float("inf")
    moves = get_possible_moves(board, color)
    utilities = []

    for move in moves:
        successor_state = play_move(board, color, move[0], move[1])
        u = alphabeta_min_node(successor_state, color, alpha, beta, limit, caching, ordering)[1]
        alpha = max(alpha, u)
        utilities.append(alpha)

    optimal_action = moves[utilities.index(max(utilities))]

    return optimal_action


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
