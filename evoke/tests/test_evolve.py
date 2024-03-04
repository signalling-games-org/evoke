# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:48:01 2024

@author: stephenfmann

Unit tests for evoke/src/evolve.py
"""

import unittest
import numpy as np

import evoke.src.evolve as evolve
import evoke.src.asymmetric_games as asy
import evoke.src.symmetric_games as sym


class TestEvolve(unittest.TestCase):
    def test_OnePop(self):
        """
        Test OnePop class

        Returns
        -------
        None.

        """

        # Just copy payoffs from a Skyrms figure
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
                [0, 0.5, 0.5, 1],
            ]  # Payoffs received by type 4 when encountering types 1, 2, 3, 4
        )

        # Create simple NoSignal game
        game = sym.NoSignal(payoffs)

        # Create the playertypes
        playertypes = np.array(
            [
                [1, 0, 0, 0],  # "I'm playing the first strategy!"
                [0, 1, 0, 0],  # "I'm playing the second strategy!"
                [0, 0, 1, 0],  # "I'm playing the third strategy!"
                [0, 0, 0, 1],
            ]  # "I'm playing the fourth strategy!"
        )

        ## ...and the simulation.
        evo = evolve.OnePop(game, playertypes)

        # Call all the methods
        self.assertEqual(evo.avg_payoff(np.array([1 / 2, 1 / 2, 0, 0])), 1 / 2)

        # Just check these don't crash
        evo.avg_payoff_vector(np.array([1, 0, 0, 0]))
        evo.discrete_replicator_delta_X(np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]))
        evo.pop_to_mixed_strat(np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]))
        evo.random_player()
        evo.replicator_dX_dt_odeint(np.array([0, 1 / 3, 1 / 3, 1 / 3]), 0)
        evo.replicator_discrete(np.array([1 / 8, 1 / 4, 1 / 4, 1 / 8]), 4)
        evo.replicator_jacobian_odeint(np.array([0, 1 / 3, 1 / 3, 1 / 3]))
        evo.replicator_odeint(
            np.array([6 / 16, 1 / 16, 4 / 16, 5 / 16]), range(10)
        )  # 10 time steps

        # Check is instance
        self.assertIsInstance(evo, evolve.OnePop)

    def test_TwoPops(self):
        """
        Test TwoPops class

        Returns
        -------
        None.

        """

        # Copy parameters from Skyrms example
        states = np.array([0.5, 0.5])
        sender_payoff_matrix = np.eye(2)
        receiver_payoff_matrix = np.eye(2)
        messages = 2

        ## Create the game
        lewis22 = asy.Chance(
            states,
            sender_payoff_matrix,
            receiver_payoff_matrix,
            messages,
        )

        ## Just get the pure strategies.
        sender_strats = lewis22.sender_pure_strats()[1:-1]
        receiver_strats = lewis22.receiver_pure_strats()[1:-1]

        ## Create the two-population simulation object
        evo = evolve.TwoPops(lewis22, sender_strats, receiver_strats)

        # Check methods with random population distributions.
        random_sender = evo.random_sender()  # a vector
        random_receiver = evo.random_receiver()  # a vector
        random_population = np.hstack((random_sender, random_receiver))
        # This method expects a 1D array that just concatenates the sender pop
        # and the receiver pop.

        evo.discrete_replicator_delta_X(random_population)
        evo.receiver_avg_payoff(random_sender, random_receiver)
        evo.receiver_to_mixed_strat(random_receiver)
        evo.replicator_dX_dt_ode(t=0, X=random_population)
        evo.replicator_dX_dt_odeint(X=random_population, t=0)
        evo.replicator_jacobian_ode(t=0, X=random_population)
        evo.replicator_jacobian_odeint(random_population)
        evo.sender_avg_payoff(random_sender, random_receiver)
        evo.sender_to_mixed_strat(random_sender)
        evo.vector_to_populations(random_population)

        # There is no longer game.Times object, so we can't test these methods
        # evo.replicator_discrete(sinit, rinit, times)
        # evo.replicator_ode(sinit, rinit, times)
        # evo.replicator_odeint(sinit, rinit, times)

        # Check is instance
        self.assertIsInstance(evo, evolve.TwoPops)


if __name__ == "__main__":
    unittest.main()
