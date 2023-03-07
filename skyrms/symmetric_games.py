"""
Set up a symmetric evolutionary game, that can be then fed to the evolve
module. For the time being, these are just games without a chance player
(although there could be games with *two* chance players).
"""
import itertools as it
import sys

import numpy as np
np.set_printoptions(precision=4)


class NoSignal:
    """
    Construct a payoff function for a game without chance player: and in which
    no one signals. Both players have the same payoff matrix
    """
    def __init__(self, payoff_matrix):
        """
        Take a square numpy array with player payoffs
        """
        if payoff_matrix.shape[0] != payoff_matrix.shape[1]:
            sys.exit("Payoff matrix should be square")
        self.chance_node = False  # flag to know where the game comes from
        self.payoff_matrix = payoff_matrix
        self.states = payoff_matrix.shape[0]

    def pure_strats(self):
        """
        Return the set of pure strategies available to the players. For this
        sort of games, a strategy is a probablity vector over the set of states
        """
        return np.eye(self.states)

    def payoff(self, first_player, second_player):
        """
        Calculate the average payoff for first and second given concrete
        strats
        """
        payoff = first_player @ self.payoff_matrix @ second_player
        return payoff

    def avg_payoffs(self, player_strats):
        """
        Return an array with the average payoff of strat i against
        strat j in position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(player_strats[int(i)],
                                                          player_strats[int(j)]))
        shape_result = [len(player_strats)] * 2
        return np.fromfunction(payoff_ij, shape_result)

    def calculate_mixed_strat(self, types, pop):
        return types @ pop


class BothSignal:
    """
    Construct a payoff function for a game without chance player, and in which
    both players signal before acting: players one and two choose a message
    among m possible ones; then they both choose acts among o possible ones.
    """
    def __init__(self, payoff_matrix, msgs):
        """
        Take a square numpy array with the payoffs, and the number of available
        messages 
        """
        if payoff_matrix.shape[0] != payoff_matrix.shape[1]:
            sys.exit("Payoff matrix should be square")
        if not isinstance(msgs, int):
            sys.exit("The number of messages for the player receiver should "
                     "be an integer")
        self.chance_node = False  # flag to know where the game comes from
        self.both_signal = True  # ... and another flag (this needs fixing)
        self.payoff_matrix = payoff_matrix
        self.states = payoff_matrix.shape[0]
        self.msgs = msgs

    def pure_strats(self):
        """
        Return the set of pure strategies available to the player. For this
        sort of games, a strategy is an mxmxo matrix in which the only non-zero
        row, r,  gives the state to be assumed in the presence of sender
        message r, and receiver message given by the column
        """
        def build_strat(msg, row):
            zeros = np.zeros((self.msgs, self.msgs, self.states))
            zeros[msg] = row
            return zeros
        states = np.identity(self.states)
        rows = np.array([row for row in it.product(states, repeat=self.msgs)])
        return np.array([build_strat(message, row) for message, row in
                         it.product(range(self.msgs), rows)])

    def payoff(self, first_player, second_player):
        """
        Calculate the average payoff for the first player given concrete first
        and second player strats
        """
        two_states = np.tensordot(first_player, second_player, axes=([0, 1],
                                                                     [1, 0]))
        payoff = np.sum(two_states * self.payoff_matrix)
        return payoff

    def avg_payoffs(self, player_strats):
        """
        Return an array with the average payoff of strat i against strat j in
        position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(player_strats[i],
                                                          player_strats[j]))
        shape_result = [len(player_strats)] * 2
        return np.fromfunction(payoff_ij, shape_result)

    def calculate_mixed_strat(self, types, pop):
        mixedstrat = types * pop[:, np.newaxis, np.newaxis, np.newaxis]
        return sum(mixedstrat)
