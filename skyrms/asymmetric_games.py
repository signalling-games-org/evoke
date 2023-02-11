"""
Set up an asymmetric evolutionary game, that can be then fed to the evolve
module.  There are two main classes here:
    - Games with a chance player
    - Games without a chance player
"""
import itertools as it
import sys

import numpy as np
np.set_printoptions(precision=4)


class Chance:
    """
    Construct a payoff function for a game with a chance player, that chooses a
    state, among m possible ones; a sender that chooses a message, among n
    possible ones; a receiver that chooses an act among o possible ones; and
    the number of messages
    """
    def __init__(self, state_chances, sender_payoff_matrix,
                 receiver_payoff_matrix, messages):
        """
        Take a mx1 numpy array with the unconditional probabilities of states,
        a mxo numpy array with the sender payoffs, a mxo numpy array with
        receiver payoffs, and the number of available messages
        """
        if any(state_chances.shape[0] != row for row in
               [sender_payoff_matrix.shape[0],
                receiver_payoff_matrix.shape[0]]):
            sys.exit("The number of rows in sender and receiver payoffs should"
                     "be the same as the number of states")
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same"
                     "shape")
        if not isinstance(messages, int):
            sys.exit("The number of messages should be an integer")
        self.state_chances = state_chances
        self.sender_payoff_matrix = sender_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.states = state_chances.shape[0]
        self.messages = messages
        self.acts = sender_payoff_matrix.shape[1]

    def sender_pure_strats(self):
        """
        Return the set of pure strategies available to the sender
        """
        pure_strats = np.identity(self.messages)
        return np.array(list(it.product(pure_strats, repeat=self.states)))

    def receiver_pure_strats(self):
        """
        Return the set of pure strategies available to the receiver
        """
        pure_strats = np.identity(self.acts)
        return np.array(list(it.product(pure_strats, repeat=self.messages)))

    def one_pop_pure_strats(self):
        """
        Return the set of pure strategies available to players in a
        one-population setup
        """
        return player_pure_strats(self)

    def payoff(self, sender_strat, receiver_strat):
        """
        Calculate the average payoff for sender and receiver given concrete
        sender and receiver strats
        """
        state_act = sender_strat.dot(receiver_strat)
        sender_payoff = self.state_chances.dot(np.sum(state_act *
                                               self.sender_payoff_matrix,
                                               axis=1))
        receiver_payoff = self.state_chances.dot(np.sum(state_act *
                                                 self.receiver_payoff_matrix,
                                                 axis=1))
        return (sender_payoff, receiver_payoff)

    def avg_payoffs(self, sender_strats, receiver_strats):
        """
        Return an array with the average payoff of sender strat i against
        receiver strat j in position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(sender_strats[i],
                                                          receiver_strats[j]))
        shape_result = (sender_strats.shape[0], receiver_strats.shape[0])
        return np.fromfunction(payoff_ij, shape_result, dtype=int)

    def one_pop_avg_payoffs(self, one_player_strats):
        """
        Return an array with the average payoff of one-pop strat i against
        one-pop strat j in position <i, j>
        """
        return one_pop_avg_payoffs(self, one_player_strats)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calculate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis,
                                                         np.newaxis]
        return sum(mixedstratreceiver)

class ChanceSIR:
    """
        A sender-intermediary-receiver game with Nature choosing the state.
    """
    
    def __init__(self,
            state_chances               = np.array([1/2,1/2]),
            sender_payoff_matrix        = np.eye(2),
            intermediary_payoff_matrix  = np.eye(2),
            receiver_payoff_matrix      = np.eye(2), 
            messages_sender             = 2,
            messages_intermediary       = 2
            ):
        """
        For now, just load the inputs into class attributes.
        We can add methods to this as required.

        Parameters
        ----------
        state_chances : TYPE, optional
            DESCRIPTION. The default is np.array([1/2,1/2]).
        sender_payoff_matrix : TYPE, optional
            DESCRIPTION. The default is np.eye(2).
        intermediary_payoff_matrix : TYPE, optional
            DESCRIPTION. The default is np.eye(2).
        receiver_payoff_matrix : TYPE, optional
            DESCRIPTION. The default is np.eye(2).
        messages_sender : TYPE, optional
            DESCRIPTION. The default is 2.
        messages_intermediary : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        
        self.state_chances              = state_chances
        self.sender_payoff_matrix       = sender_payoff_matrix
        self.intermediary_payoff_matrix = intermediary_payoff_matrix
        self.receiver_payoff_matrix     = receiver_payoff_matrix
        self.messages_sender            = messages_sender
        self.messages_intermediary      = messages_intermediary
        
        ## Call it a "regular" game if payoffs are all np.eye(2)
        self.regular = False
        
        if np.eye(2).all()                      ==\
        self.sender_payoff_matrix.all()         ==\
        self.intermediary_payoff_matrix.all()   ==\
        self.receiver_payoff_matrix.all():
            self.regular = True
        
    
    def choose_state(self):
        """
        Randomly get a state according to self.state_chances

        Returns
        -------
        state : int
            Index of the chosen state.

        """
        
        return np.random.choice(range(len(self.state_chances)),p=self.state_chances)
    
    def payoff_sender(self,state,act):
        """
        Get the sender's payoff when this combination of state and act occurs.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        act : TYPE
            DESCRIPTION.

        Returns
        -------
        payoff : TYPE
            DESCRIPTION.

        """
        
        return self.sender_payoff_matrix[state][act]
    
    def payoff_intermediary(self,state,act):
        """
        Get the intermediary's payoff when this combination of state and act occurs.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        act : TYPE
            DESCRIPTION.

        Returns
        -------
        payoff : TYPE
            DESCRIPTION.

        """
        
        return self.intermediary_payoff_matrix[state][act]
    
    def payoff_receiver(self,state,act):
        """
        Get the receiver's payoff when this combination of state and act occurs.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        act : TYPE
            DESCRIPTION.

        Returns
        -------
        payoff : TYPE
            DESCRIPTION.

        """
        
        return self.receiver_payoff_matrix[state][act]
    
    def avg_payoffs_regular(self, 
                            sender_strats, 
                            intermediary_strats,
                            receiver_strats):
        """
        Return the average payoff of all players given these strategy profiles.
        
        Requires game to be regular.

        Parameters
        ----------
        sender_strats : TYPE
            DESCRIPTION.
        intermediary_strats: array-like
            Intermediary player's strategy profile, normalised
        receiver_strats : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        assert self.regular
        
        ## Nature and Sender combine to provide unconditional probabilities
        ##  of sender signals 0 and 1.
        prob_uncond_signal_sender = self.state_chances @ sender_strats
        
        ## Sender signals and intermediary combine to provide unconditional probabilities
        ##  of intermediary signals 0 and 1.
        prob_uncond_signal_intermediary = prob_uncond_signal_sender @ intermediary_strats
        
        ## Intermediary signals and receiver combine to provide unconditional probabilities
        ##  of acts.
        prob_uncond_acts = prob_uncond_signal_intermediary @ receiver_strats
        
        ## TODO
        

class NonChance:
    """
    Construct a payoff function for a game without chance player: a sender that
    chooses a message, among n possible ones; a receiver that chooses an act
    among o possible ones; and the number of messages
    """
    def __init__(self, sender_payoff_matrix, receiver_payoff_matrix, messages):
        """
        Take a mxo numpy array with the sender payoffs, a mxo numpy array
        with receiver payoffs, and the number of available messages
        """
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same"
                     "shape")
        if not isinstance(messages, int):
            sys.exit("The number of messages should be an integer")
        self.chance_node = False  # flag to know where the game comes from
        self.sender_payoff_matrix = sender_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.states = sender_payoff_matrix.shape[0]
        self.messages = messages
        self.acts = sender_payoff_matrix.shape[1]

    def sender_pure_strats(self):
        """
        Return the set of pure strategies available to the sender. For this
        sort of games, a strategy is a tuple of vector with probability 1 for
        the sender's state, and an mxn matrix in which the only nonzero row
        is the one that correspond's to the sender's type.
        """
        def build_strat(state, row):
            zeros = np.zeros((self.states, self.messages))
            zeros[state] = row
            return zeros
        states = range(self.states)
        over_messages = np.identity(self.messages)
        return np.array([build_strat(state, row) for state, row in
                         it.product(states, over_messages)])

    def receiver_pure_strats(self):
        """
        Return the set of pure strategies available to the receiver
        """
        pure_strats = np.identity(self.acts)
        return np.array(list(it.product(pure_strats, repeat=self.messages)))

    def one_pop_pure_strats(self):
        """
        Return the set of pure strategies available to players in a
        one-population setup
        """
        return player_pure_strats(self)

    def payoff(self, sender_strat, receiver_strat):
        """
        Calculate the average payoff for sender and receiver given concrete
        sender and receiver strats
        """
        state_act = sender_strat.dot(receiver_strat)
        sender_payoff = np.sum(state_act * self.sender_payoff_matrix)
        receiver_payoff = np.sum(state_act * self.receiver_payoff_matrix)
        return (sender_payoff, receiver_payoff)

    def avg_payoffs(self, sender_strats, receiver_strats):
        """
        Return an array with the average payoff of sender strat i against
        receiver strat j in position <i, j>
        """
        payoff_ij = np.vectorize(lambda i, j: self.payoff(sender_strats[i],
                                                          receiver_strats[j]))
        shape_result = (len(sender_strats), len(receiver_strats))
        return np.fromfunction(payoff_ij, shape_result)

    def one_pop_avg_payoffs(self, one_player_strats):
        """
        Return an array with the average payoff of one-pop strat i against
        one-pop strat j in position <i, j>
        """
        return one_pop_avg_payoffs(self, one_player_strats)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calculate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis,
                                                         np.newaxis]
        return sum(mixedstratreceiver)
