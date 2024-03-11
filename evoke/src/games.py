# -*- coding: utf-8 -*-
"""
Set up an asymmetric evolutionary game, that can be then fed to the evolve
module.  There are two main classes here:
    
- ``Chance``: Games with a chance player
- ``NonChance``: Games without a chance player
"""
import itertools as it

import numpy as np

# import pygambit

# Optional modules
try:
    import pygambit

    PYGAMBIT_EXISTS = True
except ModuleNotFoundError:
    PYGAMBIT_EXISTS = False

from evoke.src import info
from evoke.src import exceptions as ex

np.set_printoptions(precision=4)

# Significant figures for certain calculations.
SIGFIG = 5


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

        # Check data is consistent
        if any(
            state_chances.shape[0] != row
            for row in [sender_payoff_matrix.shape[0], receiver_payoff_matrix.shape[0]]
        ):
            exception_message = (
                "The number of rows in sender and receiver payoffs should "
                + "be the same as the number of states"
            )
            raise ex.InconsistentDataException(exception_message)

        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            raise ex.InconsistentDataException(
                "Sender and receiver payoff arrays should have the same shape"
            )

        if not isinstance(messages, int) or messages < 1:
            raise ex.InconsistentDataException(
                "The number of messages should be a positive integer"
            )

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

    # def one_pop_avg_payoffs(self, one_player_strats):
    #     """
    #     Return an array with the average payoff of one-pop strat i against
    #     one-pop strat j in position <i, j>
    #     """
    #     return one_pop_avg_payoffs(self, one_player_strats)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calculate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratreceiver)

    def create_gambit_game(self):
        """
        Create a gambit object based on this game.

        [SFM: UPDATE: this method has changed significantly
        to comply with pygambit 16.1.0.
        Original note: For guidance creating this method I followed the tutorial at
        https://nbviewer.org/github/gambitproject/gambit/blob/master/contrib/samples/sendrecv.ipynb
        and adapted as appropriate.]

        Returns
        -------
        g: Game() object from pygambit package.
        """

        # pygambit must exist
        if not PYGAMBIT_EXISTS:
            raise ex.ModuleNotInstalledException(
                "ERROR: This method requires pygambit 16.1.0 or higher, which is not installed on this system."
            )

        # Lazy instantiation
        if hasattr(self, "_g"):
            return self._g

        ## Initialize.
        ## Game.new_tree() creates a new, trivial extensive game,
        ##  with no players, and only a root node.
        g = pygambit.Game.new_tree()

        ## Game title
        g.title = (
            f"Chance Sender-Receiver game {self.states}x{self.messages}x{self.acts}"
        )

        ## Players: Sender and Receiver
        ## There is already a built-in chance player at game_gambit.players.chance
        sender = g.add_player("Sender")
        receiver = g.add_player("Receiver")

        ## Add Nature's initial move
        # move_nature = g.root.append_move(g.players.chance, self.states)
        g.append_move(
            node=g.root,
            player=g.players.chance,
            actions=np.arange(self.states).astype(str).tolist(),
        )
        move_nature = g.root.children

        ## STRATEGIES
        ## Label Nature's possible actions, and add the sender's response.
        moves_receiver = []
        for i in range(self.states):
            ## Label the state from its index.
            move_nature[i].label = str(i)

            # For each state, the sender has {self.messages} actions.
            g.append_move(
                node=move_nature[i],
                player=sender,
                actions=np.arange(self.messages).astype(str).tolist(),
            )

            # The sender's move is the list of its actions
            move_sender = move_nature[i].children

            ## Label each signal with its index, and add the receiver's response.
            for j in range(self.messages):
                if i == 0:
                    move_sender[j].label = str(j)

                    g.append_move(
                        node=move_sender[j],
                        player=receiver,
                        actions=np.arange(self.acts).astype(str).tolist(),
                    )

                    # moves_receiver will be a list of lists.
                    # The j'th list contains the receiver's actions after the j'th message.
                    moves_receiver.append(move_sender[j].children)

                    for k in range(self.acts):
                        # Set the move label
                        moves_receiver[j][k].label = str(k)

                        # Set the infoset label
                        # We'll append further receiver acts to this infoset.
                        # i.e. for the same value of j (messages) but different values of i (states).
                        # That's because the receiver only knows about the message, not the state.
                        moves_receiver[j][k].prior_action.infoset.label = str(k)

                else:
                    # The j'th entry of <moves_receiver> is a list
                    # containing the receiver's actions after the j'th message.
                    # Therefore, we append <moves_receiver[j]> to <move_sender[j]>.
                    g.append_infoset(
                        node=move_sender[j],
                        infoset=moves_receiver[j][0].prior_action.infoset,
                    )

        ## OUTCOMES
        ## The size of the payoff matrices, which should be states x acts,
        ##  determines the number of outcomes.
        for row_index in range(len(self.sender_payoff_matrix)):
            for col_index in range(len(self.sender_payoff_matrix[row_index])):
                ## Create the outcome.
                outcome_label = f"payoff_{row_index}_{col_index}"

                ## Sender's payoff at this outcome.
                ## Gambit only accepts integer payoffs!!!
                if (
                    int(self.sender_payoff_matrix[row_index][col_index])
                    != self.sender_payoff_matrix[row_index][col_index]
                ):
                    print(
                        f"Warning: converting payoff {self.sender_payoff_matrix[row_index][col_index]} to integer"
                    )

                # get sender's payoff at this outcome, as an integer
                sender_payoff_outcome = int(
                    self.sender_payoff_matrix[row_index][col_index]
                )

                ## Receiver's payoff at this outcome
                ## Gambit only accepts integer payoffs!!!
                if (
                    int(self.receiver_payoff_matrix[row_index][col_index])
                    != self.receiver_payoff_matrix[row_index][col_index]
                ):
                    print(
                        f"Warning: converting payoff {self.receiver_payoff_matrix[row_index][col_index]} to integer"
                    )

                # get receiver's payoff at this outcome, as an integer
                receiver_payoff_outcome = int(
                    self.receiver_payoff_matrix[row_index][col_index]
                )

                # Create outcome
                outcome = g.add_outcome(
                    [sender_payoff_outcome, receiver_payoff_outcome],
                    label=outcome_label,
                )

                ## Append this outcome to the game across all the different
                ##  possible signals that could lead to it.
                for j in range(self.messages):
                    g.set_outcome(
                        node=g.root.children[row_index].children[j].children[col_index],
                        outcome=outcome,
                    )

        ## Return the game object.
        self._g = g
        return self._g

    @property
    def has_info_using_equilibrium(self) -> bool:
        """

        Does this game have an information-using equilibrium?


        Parameters
        ----------
        sigfig : int, optional
            The number of significant figures to report values in.
            Since gambit sometimes has problems rounding, it generates values like 0.9999999999996.
            We want to report these as 1.0000, especially if we're dumping to a file.
            The default is 5.


        Returns
        -------
        bool
            If True, the game has at least one information-using equilibrium.
            If False, the game does not have an information-using equilibrium.

        """

        # Lazy instantiation
        if hasattr(self, "_has_info_using_equilibrium"):
            return self._has_info_using_equilibrium

        # Is there an info-using equilibrium?
        # First get the gambit game object
        gambit_game = self.create_gambit_game()

        # Now get the equilibria
        equilibria_gambit = pygambit.nash.lcp_solve(gambit_game, rational=False)

        # Convert to python list
        equilibria = eval(str(equilibria_gambit))

        # Now for each equilibrium, check whether it is info-using
        for equilibrium in equilibria:
            # Figure out whether the strategies at this equilibrium
            # lead to an information-using situation.

            # Sometimes gambit gives back long decimals e.g. 0.999999996
            # We want to round these before inspecting.
            sender_strategy = np.around(np.array(equilibrium[0]), SIGFIG)
            receiver_strategy = np.around(np.array(equilibrium[1]), SIGFIG)

            # Create info object to make info measurements
            info_object = info.Information(self, sender_strategy, receiver_strategy)

            # Get the mutual information between states and acts at this equilibrium.
            # If it's greater than zero, we've got what we need and can return.
            # Otherwise the loop will continue checking subsequent equilibria.
            if info_object.mutual_info_states_acts() > 0:
                self._has_info_using_equilibrium = True
                return self._has_info_using_equilibrium

        # No information-using equilibria were found.
        self._has_info_using_equilibrium = False
        return self._has_info_using_equilibrium

    @has_info_using_equilibrium.setter
    def has_info_using_equilibrium(self, has_info_using_equilibrium) -> None:
        """
        Manually set the boolean whether this game has an information-using equilibrium.
        Useful when you are loading the game from a file and don't want to have to
        run all the gambit processing again.

        Parameters
        ----------
        has_info_using_equilibrium : bool
            If True, this game has an information-using equilibrium.
            If False, it doesn't.

        Returns
        -------
        None.

        """

        self._has_info_using_equilibrium = has_info_using_equilibrium

    @property
    def highest_info_using_equilibrium(self) -> tuple:
        """
        Get the mutual information between states and acts at the
        equilibrium with the highest such value.
        Also get the strategies at this equilibrium

        Note that if the game has no information-using equilibria,
        the value of mutual information will be 0.
        The strategies returned will then be an arbitrary equilibrium.

        Parameters
        ----------
        sigfig : int, optional
            The number of significant figures to report values in.
            Since gambit sometimes has problems rounding, it generates values like 0.9999999999996.
            We want to report these as 1.0000, especially if we're dumping to a file.
            The default is 5.

        Returns
        -------
        tuple
            First element is a list containing the highest-info-using sender and receiver strategies.
            Second element is the mutual information between states and acts given these strategies.

        """

        # Lazy instantiation
        if hasattr(self, "_best_strategies") and hasattr(self, "_max_mutual_info"):
            return self._best_strategies, self._max_mutual_info

        # First get the gambit game object
        gambit_game = self.create_gambit_game()

        # Now get the equilibria
        equilibria_gambit = pygambit.nash.lcp_solve(gambit_game, rational=False)

        # Convert to python list
        equilibria = eval(str(equilibria_gambit))

        # Initialize
        current_highest_info_at_equilibrium = 0

        # Now for each equilibrium, check if it is info-using
        for equilibrium in equilibria:
            # Figure out whether the strategies at this equilibrium
            # lead to an information-using situation.

            # Sometimes gambit gives back long decimals e.g. 0.999999996
            # We want to round these before dumping to a file.
            sender_strategy = np.around(np.array(equilibrium[0]), SIGFIG)
            receiver_strategy = np.around(np.array(equilibrium[1]), SIGFIG)

            # Create info object to make info measurements
            info_object = info.Information(self, sender_strategy, receiver_strategy)

            # Get current mutual info
            # Sometimes it spits out -0.0, which should be 0.0.
            current_mutual_info = abs(info_object.mutual_info_states_acts())

            if current_mutual_info >= current_highest_info_at_equilibrium:
                # Update game details
                # The equilibrium is just the current sender strategy
                # followed by the current receiver strategy.
                current_best_strategies = [
                    sender_strategy.tolist(),
                    receiver_strategy.tolist(),
                ]

                # The mutual information is the current mutual info at this equilibrium.
                current_highest_info_at_equilibrium = round(current_mutual_info, SIGFIG)

        # Return the best strategies and highest info found
        self._best_strategies = current_best_strategies
        self._max_mutual_info = current_highest_info_at_equilibrium

        # If it's non-zero, set the relevant boolean attribute
        if self._max_mutual_info > 0:
            self.has_info_using_equilibrium = True
        else:
            self.has_info_using_equilibrium = False

        return self._best_strategies, self._max_mutual_info

    @highest_info_using_equilibrium.setter
    def highest_info_using_equilibrium(self, equilibrium_data) -> None:
        """
        Manually set the best strategy and amount of mutual information between
        states and acts at the highest information-using equilibrium.
        Useful when you are loading the game from a file and don't want to have to
        run all the gambit processing again.


        Parameters
        ----------
        best_strategies : array-like
            First element is sender strat, second element is receiver strat.
        max_mutual_info : float
            The amount of mutual information between states and acts
            at the highest info-using equilibrium.

        Returns
        -------
        None

        """

        try:
            best_strategies, max_mutual_info = equilibrium_data
        except ValueError:
            exception_message = (
                "This function needs an iterable as input. "
                + "The first element should be a list [sender_strategy, receiver_strategy]. "
                + "The second element should be a float."
            )
            raise ValueError(exception_message)

        self._best_strategies = best_strategies
        self.max_mutual_info = max_mutual_info

        # If the highest info is greater than zero, also set the relevant boolean attribute.
        if self.max_mutual_info > 0:
            self.has_info_using_equilibrium = True
        else:
            self.has_info_using_equilibrium = False

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
        if not hasattr(self, "_max_mutual_info"):
            ## The maximum mutual information occurs when the conditional probabilities
            ##  of acts given states are as uneven as possible.
            ## So let's construct a fake conditional distribution of acts given states,
            ##  and set as many of its rows to one-hot vectors as possible.
            ## Then calculate the mutual information implied by that conditional distribution
            ##  together with self.state_chances.

            ## First, initialize a square identity matrix.
            maximising_conditional_distribution_states_acts = np.eye(
                min(self.states, self.acts)
            )

            ## If there are more states than acts, stack more identity matrices
            ##  up underneath the first.
            if self.states > self.acts:
                ## How many rows do we need to add?
                rows_left = self.states - self.acts

                ## Add them one by one (it could be done more cleverly than this.)
                column_to_insert_1 = 0
                while rows_left > 0:
                    ## Initialise row of length self.acts.
                    new_row = np.zeros((1, self.acts))

                    ## Insert a 1 in the correct place.
                    new_row.put(column_to_insert_1, 1)

                    ## Stack this row under the distribution.
                    maximising_conditional_distribution_states_acts = np.vstack(
                        (maximising_conditional_distribution_states_acts, new_row)
                    )

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
                horizontal_stacker = np.zeros((self.states, self.acts - self.states))

                maximising_conditional_distribution_states_acts = np.hstack(
                    (
                        maximising_conditional_distribution_states_acts,
                        horizontal_stacker,
                    )
                )

            ## Now calculate the joint of this and the states.
            maximising_joint_states_acts = info.from_conditional_to_joint(
                self.state_chances, maximising_conditional_distribution_states_acts
            )

            ## Finally, calculate the mutual information from the joint distribution.
            self._max_mutual_info = info.mutual_info_from_joint(
                maximising_joint_states_acts
            )

        return self._max_mutual_info

    @max_mutual_info.setter
    def max_mutual_info(self, max_mutual_info):
        """
        Set the maximum mutual information at equilibrium for this game.

        Parameters
        ----------
        max_mutual_info : float
            Maximum mutual information at equilibrium.
            If self.has_info_using_equilibrium is False, this should be 0.
            If self.has_info_using_equilibrium is True, this should be > 0.

        Returns
        -------
        None.

        """

        self._max_mutual_info = max_mutual_info

    @max_mutual_info.deleter
    def max_mutual_info(self):
        """
        Delete the maximum mutual information variable.

        Returns
        -------
        None.

        """

        if hasattr(self, "_max_mutual_info"):
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

        # Check data is consistent
        if any(
            state_chances.shape[0] != row
            for row in [
                sender_payoff_matrix.shape[0],
                intermediary_payoff_matrix.shape[0],
                receiver_payoff_matrix.shape[0],
            ]
        ):
            exception_message = (
                "The number of rows in sender, intermediary and receiver payoffs should "
                + "be the same as the number of states"
            )
            raise ex.InconsistentDataException(exception_message)

        if (
            sender_payoff_matrix.shape != receiver_payoff_matrix.shape
            or sender_payoff_matrix.shape != intermediary_payoff_matrix.shape
        ):
            raise ex.InconsistentDataException(
                "All agents' payoff arrays should have the same shape"
            )

        if not isinstance(messages_sender, int) or messages_sender < 1:
            raise ex.InconsistentDataException(
                "The number of sender messages should be a positive integer"
            )

        if not isinstance(messages_intermediary, int) or messages_intermediary < 1:
            raise ex.InconsistentDataException(
                "The number of intermediary messages should be a positive integer"
            )

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

        # Check data is consistent
        if sender_payoff_matrix.shape != receiver_payoff_matrix.shape:
            raise ex.InconsistentDataException(
                "Sender and receiver payoff arrays should have the same shape"
            )

        if not isinstance(messages, int) or messages < 1:
            raise ex.InconsistentDataException(
                "The number of messages should be a positive integer"
            )

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

    # def one_pop_pure_strats(self):
    #     """
    #     Return the set of pure strategies available to players in a
    #     one-population setup
    #     """
    #     return player_pure_strats(self)

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
        return np.fromfunction(payoff_ij, shape_result, dtype=int)

    # def one_pop_avg_payoffs(self, one_player_strats):
    #     """
    #     Return an array with the average payoff of one-pop strat i against
    #     one-pop strat j in position <i, j>
    #     """
    #     return one_pop_avg_payoffs(self, one_player_strats)

    def calculate_sender_mixed_strat(self, sendertypes, senderpop):
        mixedstratsender = sendertypes * senderpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratsender)

    def calculate_receiver_mixed_strat(self, receivertypes, receiverpop):
        mixedstratreceiver = receivertypes * receiverpop[:, np.newaxis, np.newaxis]
        return sum(mixedstratreceiver)

    def create_gambit_game(self):
        """
        Create a gambit object based on this game.

        [SFM: UPDATE: this method has changed significantly
        to comply with pygambit 16.1.0.
        Original note: For guidance creating this method I followed the tutorial at
        https://nbviewer.org/github/gambitproject/gambit/blob/master/contrib/samples/sendrecv.ipynb
        and adapted as appropriate.]

        Returns
        -------
        g: Game() object from pygambit package.

        """

        # pygambit must exist
        if not PYGAMBIT_EXISTS:
            raise ex.ModuleNotInstalledException(
                "ERROR: This method requires pygambit 16.1.0 or higher, which is not installed on this system."
            )

        # Lazy instantiation
        if hasattr(self, "_g"):
            return self._g

        ## Initialize.
        ## Game.new_tree() creates a new, trivial extensive game,
        ##  with no players, and only a root node.
        g = pygambit.Game.new_tree()

        ## Game title
        g.title = f"Non-Chance Sender-Receiver game {self.messages}x{self.acts}"

        ## Players: Sender and Receiver
        sender = g.add_player("Sender")
        receiver = g.add_player("Receiver")
        # sender = g.players.add("Sender")
        # receiver = g.players.add("Receiver")

        ## Add the Sender's initial move
        g.append_move(
            node=g.root,
            player=sender,
            actions=np.arange(self.messages).astype(str).tolist(),
        )
        move_sender = g.root.children
        # move_sender = g.root.append_move(sender, self.messages)

        ## Receiver
        ## Label each signal with its index, and add the receiver's response.
        for j in range(self.messages):
            # Label the signal from its index.
            move_sender[j].label = str(j)
            # signal_label = move_sender.actions[j].label = str(j)

            ## For each signal, the receiver has {self.acts} actions.
            g.append_move(
                node=move_sender[j],
                player=receiver,
                actions=np.arange(self.acts).astype(str).tolist(),
            )

            move_receiver = move_sender[j].children
            # move_receiver = g.root.children[j].append_move(receiver, self.acts)

            ## Label the receiver's choice node at this point.
            ## All it knows about is the signal.
            # move_receiver.label = f"r{signal_label}"

            ## Label each act with its index.
            for k in range(self.acts):
                move_receiver[k].label = str(k)
                # move_receiver.actions[k].label = str(k)

        ## OUTCOMES
        ## The size of the payoff matrices, which should be messages x acts,
        ##  determines the number of outcomes.
        for row_index in range(len(self.sender_payoff_matrix)):
            for col_index in range(len(self.sender_payoff_matrix[row_index])):
                # Initialise payoff list
                outcome_payoffs = [0, 0]
                outcome_label = f"payoff_{row_index}_{col_index}"
                # outcome = g.outcomes.add(f"payoff_{row_index}_{col_index}")

                ## Sender's payoff at this outcome.
                ## Gambit only accepts integer payoffs!!!
                if (
                    int(self.sender_payoff_matrix[row_index][col_index])
                    != self.sender_payoff_matrix[row_index][col_index]
                ):
                    print(
                        f"Warning: converting payoff {self.sender_payoff_matrix[row_index][col_index]} to integer"
                    )

                # The first entry in the outcome list
                # is the sender's payoff.
                outcome_payoffs[0] = int(
                    self.sender_payoff_matrix[row_index][col_index]
                )

                ## Receiver's payoff at this outcome
                ## Gambit only accepts integer payoffs!!!
                if (
                    int(self.receiver_payoff_matrix[row_index][col_index])
                    != self.receiver_payoff_matrix[row_index][col_index]
                ):
                    print(
                        f"Warning: converting payoff {self.receiver_payoff_matrix[row_index][col_index]} to integer"
                    )

                # The second entry in the outcome list
                # is the receiver's payoff.
                outcome_payoffs[1] = int(
                    self.receiver_payoff_matrix[row_index][col_index]
                )

                ## Create the outcome.
                outcome = g.add_outcome(payoffs=outcome_payoffs, label=outcome_label)

                # Append this outcome to the game.
                # Payoffs are per message and act, right?
                g.set_outcome(
                    node=g.root.children[row_index].children[col_index], outcome=outcome
                )
                # for j in range(self.messages):

                # g.root.children[row_index].children[j].children[
                #     col_index
                # ].outcome = outcome

        ## Return the game object.
        self._g = g
        return self._g


class NoSignal:
    """
    Construct a payoff function for a game without chance player: and in which
    no one signals. Both players have the same payoff matrix
    """

    def __init__(self, payoff_matrix):
        """
        Take a square numpy array with player payoffs
        """

        # Check data is consistent
        if payoff_matrix.shape[0] != payoff_matrix.shape[1]:
            raise ex.InconsistentDataException("Payoff matrix should be square")

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
        payoff_ij = np.vectorize(
            lambda i, j: self.payoff(player_strats[i], player_strats[j])
        )
        shape_result = [len(player_strats)] * 2
        return np.fromfunction(payoff_ij, shape_result, dtype=int)

    def calculate_mixed_strat(self, types, pop):
        return types @ pop


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
        state_chances=state_chances,
        sender_payoff_matrix=sender_payoff_matrix,
        receiver_payoff_matrix=receiver_payoff_matrix,
        messages=messages,
    )

    return lewis_n


def gambit_example(n=2, export=False, fpath="tester.efg"):
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
    try:
        g = game.create_gambit_game()
    except ex.ModuleNotInstalledException as e:
        print(e)
        return

    ## Export .efg file
    if export:
        f_data = g.write("native")

        with open(fpath, "w") as f:
            f.write(f_data)

    ## Get the Nash equilibria
    ## Set rational=False to get floats rather than Rational() objects.
    solutions = pygambit.nash.lcp_solve(g, rational=False)

    print(f"Nash equilibria are {solutions}.")
