"""
Set up an asymmetric evolutionary game, that can be then fed to the evolve
module.  There are two main classes here:
    - Games with a chance player
    - Games without a chance player
"""
import itertools as it
import sys

import numpy as np
import pygambit

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
    
    def create_gambit_game(self):
        """
        Create a gambit object based on this game.
        
        [SFM: For guidance creating this method I followed the tutorial at
        https://nbviewer.org/github/gambitproject/gambit/blob/master/contrib/samples/sendrecv.ipynb
        and adapted as appropriate.]

        Returns
        -------
        game_gambit: Game() object from pygambit package.

        """
        
        ## Initialize.
        ## Game.new_tree() creates a new, trivial extensive game, 
        ##  with no players, and only a root node.
        g = pygambit.Game.new_tree()
        
        ## Game title
        g.title = f"Chance Sender-Receiver game {self.states}x{self.messages}x{self.acts}"
        
        ## Players: Sender and Receiver
        ## There is already a built-in chance player at game_gambit.players.chance
        sender = g.players.add("Sender")
        receiver = g.players.add("Receiver")
        
        ## Add Nature's initial move
        move_nature = g.root.append_move(g.players.chance, self.states)
        
        ## STRATEGIES
        ## Label Nature's possible actions, and add the sender's response.
        moves_receiver = []
        for i in range(self.states):
            
            ## Label the state from its index.
            state_label = move_nature.actions[i].label = str(i)
        
            ## For each state, the sender has {self.messages} actions.
            ## From the tutorial: "We add these actions at different information sets 
            ##  as the sender has perfect information about the state of the world".
            move_sender = g.root.children[i].append_move(sender, self.messages)
            
            ## Label the sender's choice node at this point
            move_sender.label = f's{state_label}'
            
            ## Label each signal with its index, and add the receiver's response.
            for j in range(self.messages):
                
                ## Label the signal from its index.
                signal_label = move_sender.actions[j].label = str(j)
                
                ## For each signal, the receiver has {self.acts} actions.
                ## After the first state, the moves should be appended to 
                ##  the existing infoset.
                ## That's because the receiver doesn't know anything beyond
                ##  which signal it received.
                if i == 0:
                    moves_receiver.append(g.root.children[i].children[j].append_move(receiver,self.acts))
                    
                    ## Label the receiver's choice node at this point.
                    ## All it knows about is the signal.
                    moves_receiver[j].label = f'r{signal_label}'
                    
                    ## Label each act with its index.
                    for k in range(self.acts):
                        moves_receiver[j].actions[k].label = str(k)
                    
                else:
                    
                    ## We are in a state > 0, so we have already defined the receiver's possible moves.
                    ## Append the existing move here, corresponding to the signal j.
                    g.root.children[i].children[j].append_move(moves_receiver[j])
                
                
            
        ## OUTCOMES
        ## The size of the payoff matrices, which should be states x acts,
        ##  determines the number of outcomes.
        for row_index in range(len(self.sender_payoff_matrix)):
            
            for col_index in range(len(self.sender_payoff_matrix[row_index])):
                
                ## Create the outcome.
                outcome = g.outcomes.add(f"payoff_{row_index}_{col_index}")
                
                ## Sender's payoff at this outcome.
                ## Gambit only accepts integer payoffs!!!
                if int(self.sender_payoff_matrix[row_index][col_index]) != self.sender_payoff_matrix[row_index][col_index]:
                    print(f"Warning: converting payoff {self.sender_payoff_matrix[row_index][col_index]} to integer")
                    
                outcome[0] = int(self.sender_payoff_matrix[row_index][col_index])
                
                ## Receiver's payoff at this outcome
                ## Gambit only accepts integer payoffs!!!
                if int(self.receiver_payoff_matrix[row_index][col_index]) != self.receiver_payoff_matrix[row_index][col_index]:
                    print(f"Warning: converting payoff {self.receiver_payoff_matrix[row_index][col_index]} to integer")
                    
                outcome[1] = int(self.receiver_payoff_matrix[row_index][col_index])
                
                ## Append this outcome to the game across all the different
                ##  possible signals that could lead to it.
                for j in range(self.messages): 
                    
                    g.root.children[row_index].children[j].children[col_index].outcome = outcome
        
        ## Return the game object.
        return g
        

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



def lewis_square(n=2):
    """
    Factory method to produce a cooperative nxnxn signalling game
     (what Skyrms calls a "Lewis signalling game").

    Returns
    -------
    game : Chance object.
        A nxnxn cooperative signalling game.

    """
    
    ## 1. Create state_chances, the probability distribution
    ##     over the state of the world.
    state_chances = np.ones((n,)) / n
    
    ## 2. Create sender_payoff_matrix and receiver_payoff_matrix,
    ##     which define the payoffs for sender and receiver.
    ##    In a fully cooperative game they are identical.
    sender_payoff_matrix = receiver_payoff_matrix = np.eye(n)
    
    ## 3. Define the number of messages.
    messages = n
    
    ## 4. Create the game
    lewis_n = Chance(
        state_chances = state_chances,
        sender_payoff_matrix = sender_payoff_matrix,
        receiver_payoff_matrix = receiver_payoff_matrix,
        messages = messages
        )
    
    return lewis_n

def gambit_example(n=2,export=False,fpath="tester.efg"):
    """
    Create the gambit representation of a cooperative nxnxn game
     and compute its Nash equilibria.
     
    Optionally output as an extensive-form game, which can be
     loaded into the Gambit GUI.

    Returns
    -------
    None.

    """
    
    ## Create the Chance() game object
    game = lewis_square(n=n)
    
    ## Create the gambit game object
    g = game.create_gambit_game()
    
    ## Export .efg file
    if export:
        f_data = g.write("native")
        
        with open(fpath,"w") as f:
            f.write(f_data)
    
    ## Get the Nash equilibria
    ## Set rational=False to get floats rather than Rational() objects.
    solutions = pygambit.nash.lcp_solve(g,rational=False)
    
    ## Now, what do these solutions actually mean?
    print(f"Nash equilibria are {solutions}.")
    
    return g
    
    