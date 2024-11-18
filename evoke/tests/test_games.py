# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:36:43 2024

@author: stephenfmann

Test scripts for evoke/src/games.py
"""

import unittest
import numpy as np

# Optional modules
try:
    # Check a specific version of pygambit exists
    import pygambit

    if pygambit.__version__ == "16.1.0" or pygambit.__version__ == "16.1.1":
        PYGAMBIT_EXISTS = True
    else:
        PYGAMBIT_EXISTS = False

except ModuleNotFoundError:
    PYGAMBIT_EXISTS = False

from evoke.src import games


class TestGames(unittest.TestCase):
    def test_Chance(self):
        """
        Test games.Chance class

        Returns
        -------
        None.

        """

        # Initialise
        n = np.random.randint(2, 5)

        # Create state_chances, the probability distribution
        #  over the state of the world.
        state_chances = np.ones((n,)) / n

        # Create sender_payoff_matrix and receiver_payoff_matrix,
        # which define the payoffs for sender and receiver.
        # In a fully cooperative game they are identical.
        sender_payoff_matrix = receiver_payoff_matrix = np.eye(n)

        # Define the number of messages.
        messages = n

        # Create the game
        game = games.Chance(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages=messages,
        )

        # Check methods
        # Pick random parameters
        rand_state = np.random.randint(0, n)
        # rand_signal = np.random.randint(0,n)
        rand_act = np.random.randint(0, n)

        game.choose_state()
        game.sender_payoff(rand_state, rand_act)
        game.receiver_payoff(rand_state, rand_act)
        sender_strats = game.sender_pure_strats()
        receiver_strats = game.receiver_pure_strats()

        # Pick a random sender strat and receiver strat for the next method.
        # Numpy doesn't allow picking from a 2D array directly, so pick random
        # row indices and then select from <sender_strats> and <receiver_strats>
        # using those indices.
        i = np.random.randint(len(sender_strats))
        rand_sender_strat = sender_strats[i]
        i = np.random.randint(len(receiver_strats))
        rand_receiver_strat = receiver_strats[i]
        game.payoff(rand_sender_strat, rand_receiver_strat)

        # Average payoffs
        game.avg_payoffs(sender_strats, receiver_strats)

        # Generate random population vectors for the next method.
        rand_sender_pop = np.random.random((len(sender_strats),))
        rand_sender_pop /= sum(rand_sender_pop)  # normalise

        rand_receiver_pop = np.random.random((len(receiver_strats),))
        rand_receiver_pop /= sum(rand_receiver_pop)  # normalise

        # Calculating mixed strategies
        game.calculate_sender_mixed_strat(sender_strats, rand_sender_pop)
        game.calculate_receiver_mixed_strat(receiver_strats, rand_receiver_pop)

        # Creating the gambit game
        if PYGAMBIT_EXISTS:
            game.create_gambit_game()

            # Properties that require pygambit
            game.has_info_using_equilibrium
            game.highest_info_using_equilibrium

        # Other properties
        game.max_mutual_info

        # Check is instance
        self.assertIsInstance(game, games.Chance)

    def test_ChanceSIR(self):
        """
        Test games.ChanceSIR class

        Returns
        -------
        None.

        """

        # Initialise
        n = np.random.randint(2, 5)

        # Create state_chances, the probability distribution
        #  over the state of the world.
        state_chances = np.ones((n,)) / n

        # Create sender_payoff_matrix and receiver_payoff_matrix,
        # which define the payoffs for sender and receiver.
        # In a fully cooperative game they are identical.
        sender_payoff_matrix = intermediary_payoff_matrix = receiver_payoff_matrix = (
            np.eye(n)
        )

        # Define the number of messages.
        messages_sender = n
        messages_intermediary = n

        # Create the game
        game = games.ChanceSIR(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            intermediary_payoff_matrix=intermediary_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages_sender=messages_sender,
            messages_intermediary=messages_intermediary,
        )

        # Check methods
        # Pick random parameters
        rand_state = np.random.randint(0, n)
        # rand_signal = np.random.randint(0,n)
        rand_act = np.random.randint(0, n)

        game.choose_state()
        game.payoff_sender(rand_state, rand_act)
        game.payoff_intermediary(rand_state, rand_act)
        game.payoff_receiver(rand_state, rand_act)

        # Define initial strategies
        # Ideally we would randomise these so that
        # each row is a conditional distribution (i.e normalised).
        # And of course they wouldn't necessarily all be (n,n),
        # though we've defined states, both sets of messages, and acts
        # as being of size n so they have to be (n,n) in this case.
        sender_strategies = np.ones((n, n))
        intermediary_strategies = np.ones((n, n))
        receiver_strategies = np.ones((n, n))

        # Sender strategy profile normalised
        # snorm is an array; each row is a list of conditional probabilities.
        snorm = (sender_strategies.T / sender_strategies.sum(1)).T

        # Intermediary strategy profile normalised
        # inorm is an array; each row is a list of conditional probabilities.
        inorm = (intermediary_strategies.T / intermediary_strategies.sum(1)).T

        # Receiver strategy profile normalised
        # rnorm is an array; each row is a list of conditional probabilities.
        rnorm = (receiver_strategies.T / receiver_strategies.sum(1)).T

        game.avg_payoffs_regular(snorm, inorm, rnorm)

        # Check is instance
        self.assertIsInstance(game, games.ChanceSIR)

    def test_NonChance(self):
        """
        Test games.NonChance class

        Returns
        -------
        None.

        """

        # Initialise
        n = np.random.randint(2, 5)

        # Create sender_payoff_matrix and receiver_payoff_matrix,
        # which define the payoffs for sender and receiver.
        # In a fully cooperative game they are identical.
        sender_payoff_matrix = receiver_payoff_matrix = np.eye(n)

        # Define the number of messages.
        messages = n

        # Create the game
        game = games.NonChance(
            sender_payoff_matrix=sender_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages=messages,
        )

        # Check methods
        sender_strats = game.sender_pure_strats()
        receiver_strats = game.receiver_pure_strats()

        # Pick a random sender strat and receiver strat for the next method.
        # Numpy doesn't allow picking from a 2D array directly, so pick random
        # row indices and then select from <sender_strats> and <receiver_strats>
        # using those indices.
        i = np.random.randint(len(sender_strats))
        rand_sender_strat = sender_strats[i]
        i = np.random.randint(len(receiver_strats))
        rand_receiver_strat = receiver_strats[i]
        game.payoff(rand_sender_strat, rand_receiver_strat)

        # Average payoffs
        game.avg_payoffs(sender_strats, receiver_strats)

        # Generate random population vectors for the next method.
        rand_sender_pop = np.random.random((len(sender_strats),))
        rand_sender_pop /= sum(rand_sender_pop)  # normalise

        rand_receiver_pop = np.random.random((len(receiver_strats),))
        rand_receiver_pop /= sum(rand_receiver_pop)  # normalise

        # Calculating mixed strategies
        game.calculate_sender_mixed_strat(sender_strats, rand_sender_pop)
        game.calculate_receiver_mixed_strat(receiver_strats, rand_receiver_pop)

        # Creating the gambit game
        if PYGAMBIT_EXISTS:
            game.create_gambit_game()

        # Check is instance
        self.assertIsInstance(game, games.NonChance)

    def test_NoSignal(self):
        """
        Test games.NoSignal class

        Returns
        -------
        None.

        """

        # Initialise
        payoffs = np.array(
            [
                [
                    1,
                    0.5,
                    0.5,
                    0,
                ],  # Payoffs received by type 1 when encountering types 1, 2, 3, 4
                [
                    0.5,
                    0,
                    1,
                    0.5,
                ],  # Payoffs received by type 2 when encountering types 1, 2, 3, 4
                [
                    0.5,
                    1,
                    0,
                    0.5,
                ],  # Payoffs received by type 3 when encountering types 1, 2, 3, 4
                [
                    0,
                    0.5,
                    0.5,
                    1,
                ],  # Payoffs received by type 4 when encountering types 1, 2, 3, 4
            ]
        )

        # Create game
        game = games.NoSignal(payoffs)

        # Check methods
        strats = game.pure_strats()

        # Create a random population
        rand_pop = np.random.random((len(payoffs),))
        rand_pop /= rand_pop.sum()  # normalise

        # Pick random players
        i = np.random.choice(len(strats))
        rand_1 = strats[i]
        i = np.random.choice(len(strats))
        rand_2 = strats[i]  # rand_2 might be the same as rand_1, but that's fine

        game.payoff(rand_1, rand_2)
        game.avg_payoffs(strats)
        game.calculate_mixed_strat(strats, rand_pop)

        # Check is instance
        self.assertIsInstance(game, games.NoSignal)


if __name__ == "__main__":
    unittest.main()
