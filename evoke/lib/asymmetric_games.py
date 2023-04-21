"""
Set up an asymmetric evolutionary game, that can be then fed to the evolve
module.  There are two main classes here:
    - Games with a chance player
    - Games without a chance player
"""
import itertools as it
import sys

import numpy as np

from evoke.lib import info

np.set_printoptions(precision=4)


class Chance:
    """
    Construct a payoff function for a game with a chance player, that chooses a
    state, among m possible ones; a sender that chooses a message, among n
    possible ones; a receiver that chooses an act among o possible ones; and
    the number of messages
    """

    def __init__(
        self, state_chances, sender_payoff_matrix, receiver_payoff_matrix, messages
    ):
        """
        Take a mx1 numpy array with the unconditional probabilities of states,
        a mxo numpy array with the sender payoffs, a mxo numpy array with
        receiver payoffs, and the number of available messages
        """
        if any(
            state_chances.shape[0] != row
            for row in [sender_payoff_matrix.shape[0], receiver_payoff_matrix.shape[0]]
        ):
            sys.exit(
                "The number of rows in sender and receiver payoffs should"
                "be the same as the number of states"
            )
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            sys.exit("Sender and receiver payoff arrays should have the same" "shape")
        if not isinstance(messages, int):
            sys.exit("The number of messages should be an integer")
        self.state_chances = state_chances
        self.sender_payoff_matrix = sender_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.states = state_chances.shape[0]
        self.messages = messages
        self.acts = sender_payoff_matrix.shape[1]

    def choose_state(self):
        """
        Return a random state, relying on the probabilities given by self.state_chances
        """
        return np.random.choice(range(self.states), p=self.state_chances)

    def sender_payoff(self, state, act):
        """
        Return the sender payoff for a combination of <state> and <act>
        """
        return self.sender_payoff_matrix[state][act]

    def receiver_payoff(self, state, act):
        """
        Return the receiver payoff for a combination of <state> and <act>
        """
        return self.receiver_payoff_matrix[state][act]

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
        # TODO

    def payoff(self, sender_strat, receiver_strat):
        """
        Calculate the average payoff for sender and receiver given concrete
        sender and receiver strats
        """
        state_act = sender_strat.dot(receiver_strat)
        sender_payoff = self.state_chances.dot(
            np.sum(state_act * self.sender_payoff_matrix, axis=1)
        )
        receiver_payoff = self.state_chances.dot(
            np.sum(state_act * self.receiver_payoff_matrix, axis=1)
        )
        return (sender_payoff, receiver_payoff)

    def avg_payoffs(self, sender_strats, receiver_strats):
        """
        Return an array with the average payoff of sender strat i against
        receiver strat j in position <i, j>
        """
        payoff_ij = np.vectorize(
            lambda i, j: self.payoff(sender_strats[i], receiver_strats[j])
        )
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
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratreceiver)
    
    @property
    def max_mutual_info(self):
        """
        Maximum possible mutual information between states and acts.
        Depends on self.state_chances.
        
        Lazy instantiation

        Returns
        -------
        _max_mutual_info : float
            The maximum mutual information between states and acts for this game.

        """
        
        ## Lazy instantiation: if we haven't yet calculated the maximum mutual information,
        ##  calculate it now.
        if not hasattr(self,"_max_mutual_info"):
            
            ## The maximum mutual information occurs when the conditional probabilities
            ##  of acts given states are as uneven as possible.
            ## So let's construct a fake conditional distribution of acts given states,
            ##  and set as many of its rows to one-hot vectors as possible.
            ## Then calculate the mutual information implied by that conditional distribution
            ##  together with self.state_chances.
            
            ## First, initialize a square identity matrix.
            maximising_conditional_distribution_states_acts = np.eye(min(self.states,self.acts))
            
            ## If there are more states than acts, stack more identity matrices
            ##  up underneath the first.
            if self.states > self.acts:
                
                ## How many rows do we need to add?
                rows_left = self.states - self.acts
                
                ## Add them one by one (it could be done more cleverly than this.)
                column_to_insert_1 = 0
                while rows_left > 0:
                    
                    ## Initialise row of length self.acts.
                    new_row = np.zeros((1,self.acts))
                    
                    ## Insert a 1 in the correct place.
                    new_row.put(column_to_insert_1,1)
                    
                    ## Stack this row under the distribution.
                    maximising_conditional_distribution_states_acts = np.vstack((
                        maximising_conditional_distribution_states_acts,
                        new_row
                        ))
                    
                    ## Increment where the next 1 goes
                    column_to_insert_1 += 1
                    
                    ## Loop back to the first column if necessary
                    column_to_insert_1 = column_to_insert_1 % self.acts
                    
                    ## There is now 1 fewer rows to insert.
                    rows_left -= 1
                    
        
            ## If there are more acts than states, just add a bunch of zeros
            ##  to the end of each row.
            if self.states < self.acts:
            
                ## Create a matrix of zeros, with self.states rows and
                ##  (self.acts - self.states) columns
                horizontal_stacker = np.zeros((self.states,self.acts-self.states))
                
                maximising_conditional_distribution_states_acts = np.hstack((
                    maximising_conditional_distribution_states_acts,
                    horizontal_stacker
                    ))
            
                
            ## Now calculate the joint of this and the states.
            maximising_joint_states_acts = info.from_conditional_to_joint(self.state_chances,maximising_conditional_distribution_states_acts)
            
            ## Finally, calculate the mutual information from the joint distribution.
            self._max_mutual_info = info.mutual_info_from_joint(maximising_joint_states_acts)
        
        return self._max_mutual_info
    
    @max_mutual_info.deleter
    def max_mutual_info(self):
        """
        Delete the maximum mutual information variable.

        Returns
        -------
        None.

        """
        
        if hasattr(self,"_max_mutual_info"):
            del self._max_mutual_info
        


class ChanceSIR:
    """
    A sender-intermediary-receiver game with Nature choosing the state.
    """

    def __init__(
        self,
        state_chances=np.array([1 / 2, 1 / 2]),
        sender_payoff_matrix=np.eye(2),
        intermediary_payoff_matrix=np.eye(2),
        receiver_payoff_matrix=np.eye(2),
        messages_sender=2,
        messages_intermediary=2,
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

        self.state_chances = state_chances
        self.sender_payoff_matrix = sender_payoff_matrix
        self.intermediary_payoff_matrix = intermediary_payoff_matrix
        self.receiver_payoff_matrix = receiver_payoff_matrix
        self.messages_sender = messages_sender
        self.messages_intermediary = messages_intermediary

        ## Call it a "regular" game if payoffs are all np.eye(2)
        self.regular = False

        if (
            np.eye(2).all()
            == self.sender_payoff_matrix.all()
            == self.intermediary_payoff_matrix.all()
            == self.receiver_payoff_matrix.all()
        ):
            self.regular = True

    def choose_state(self):
        """
        Randomly get a state according to self.state_chances

        Returns
        -------
        state : int
            Index of the chosen state.

        """

        return np.random.choice(range(len(self.state_chances)), p=self.state_chances)

    def payoff_sender(self, state, act):
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

    def payoff_intermediary(self, state, act):
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

    def payoff_receiver(self, state, act):
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

    def avg_payoffs_regular(self, snorm, inorm, rnorm):
        """
        Return the average payoff of all players given these strategy profiles.

        Requires game to be regular.

        Parameters
        ----------
        snorm : array-like
            Sender's strategy profile, normalised.
        inorm: array-like
            Intermediary player's strategy profile, normalised.
        rnorm : array-like
            Receiver's strategy profile, normalised.

        Returns
        -------
        payoff: float
            The average payoff given these strategies.
            The payoff is the same for every player.

        """

        assert self.regular

        ## Multiply the chain of strategy probabilities,
        ##  and then multiply the result by the (shared) payoff matrix.

        ## Step 1: We have the conditional probabilities of s-signals given states,
        ##          and the conditional probabilities of i-signals given s-signals,
        ##          and we want the conditional probabilities of i-signals given states.
        inorm_given_states = np.matmul(snorm, inorm)

        ## Step 2: We have the conditional probabilities of i-signals given states,
        ##          and the conditional probabilities of r-acts given i-signals,
        ##          and we want the conditional probabilities of r-acts given states.
        racts_given_states = np.matmul(inorm_given_states, rnorm)

        ## Step 3: We have the conditional probabilities of r-acts given states,
        ##          and the unconditional probabilities of states,
        ##          and we want the joint probabilities of states and r-acts.
        joint_states_and_acts = np.multiply(self.state_chances, racts_given_states)

        ## Step 4: We have the joint probabilities and the payoffs,
        ##          and we want the overall expected payoff.
        ## Remember all the payoff matrices are the same, so we can use any of them.
        payoff = np.multiply(joint_states_and_acts, self.sender_payoff_matrix).sum()

        return payoff


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
            sys.exit("Sender and receiver payoff arrays should have the same" "shape")
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
        return np.array(
            [
                build_strat(state, row)
                for state, row in it.product(states, over_messages)
            ]
        )

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
        payoff_ij = np.vectorize(
            lambda i, j: self.payoff(sender_strats[i], receiver_strats[j])
        )
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
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratreceiver)
