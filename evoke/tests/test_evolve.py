# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:48:01 2024

@author: stephenfmann

Unit tests for evoke/src/evolve.py
"""

import unittest
import numpy as np

import evoke.src.evolve as evolve
from evoke.src import games


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
        game = games.NoSignal(payoffs)

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
        evo.pop_to_mixed_strat(np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]))
        evo.random_player()
        evo.replicator_jacobian_odeint(np.array([0, 1 / 3, 1 / 3, 1 / 3]))

        # Check the replicator equations yield vectors that sum to 1
        self.assertEqual(
            1,
            evo.discrete_replicator_delta_X(
                np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
            ).sum(),
        )  # 1 step
        self.assertEqual(
            1,
            evo.replicator_discrete(np.array([1 / 8, 1 / 4, 1 / 4, 1 / 8]), 4)[
                -1
            ].sum(),
        )  # 4 steps
        self.assertEqual(
            1,
            evo.replicator_odeint(
                np.array([6 / 16, 1 / 16, 4 / 16, 5 / 16]), range(10)
            )[-1].sum(),
        )  # 10 steps

        # Check the continuous replicator dX/dt vector sums to 0 (or nearly so)
        self.assertAlmostEqual(
            0,
            evo.replicator_dX_dt_odeint(np.array([0, 1 / 3, 1 / 3, 1 / 3]), 0).sum(),
            5,
        )

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
        lewis22 = games.Chance(
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

        # Just check these don't crash
        evo.receiver_avg_payoff(random_sender, random_receiver)
        evo.receiver_to_mixed_strat(random_receiver)
        evo.sender_avg_payoff(random_sender, random_receiver)
        evo.sender_to_mixed_strat(random_sender)
        evo.replicator_jacobian_ode(t=0, X=random_population)
        evo.replicator_jacobian_odeint(random_population)

        # Check vector_to_populations yields an array of length 2
        self.assertEqual(2, len(evo.vector_to_populations(random_population)))

        # Check the replicator equations yield vectors that sum to 2 (that's 1 per population)
        self.assertAlmostEqual(
            2, evo.discrete_replicator_delta_X(random_population).sum(), 5
        )  # 1 step

        # Check the continuous replicator dX/dt vector sums to 0 (or nearly so)
        self.assertAlmostEqual(
            0, evo.replicator_dX_dt_ode(t=0, X=random_population).sum(), 5
        )
        self.assertAlmostEqual(
            0, evo.replicator_dX_dt_odeint(X=random_population, t=0).sum(), 5
        )

        # Create evolve.Times object to test replicator methods
        initial_time = 0
        final_time = np.random.randint(5, 20)
        time_inc = 0.05
        times = evolve.Times(initial_time, final_time, time_inc)

        # Check replicator methods yield vectors that sum to 2 (that's 1 per population)
        self.assertAlmostEqual(
            2,
            evo.replicator_discrete(random_sender, random_receiver, times)[-1].sum(),
            5,
        )
        self.assertAlmostEqual(
            2, evo.replicator_ode(random_sender, random_receiver, times)[-1].sum(), 5
        )
        self.assertAlmostEqual(
            2, evo.replicator_odeint(random_sender, random_receiver, times)[-1].sum(), 5
        )

        # Check is instance
        self.assertIsInstance(evo, evolve.TwoPops)

    def test_MatchingSR(self):
        """
        Test evolve.MatchingSR class

        Returns
        -------
        None.

        """

        # Copy parameters from Skyrms example
        p1_random = np.round(np.random.random(), 4)
        state_chances = np.array([p1_random, 1 - p1_random])
        sender_payoff_matrix = np.eye(2)
        receiver_payoff_matrix = np.eye(2)
        messages = 2

        # Create game
        game = games.Chance(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages=messages,
        )

        # Define strategies
        sender_strategies = np.ones((2, 2))
        receiver_strategies = np.ones((2, 2))

        # Create simulation
        evo = evolve.MatchingSR(game, sender_strategies, receiver_strategies)

        # Check initial stats
        evo.calculate_stats()

        # The initial statistics should be zero mutual info and 1/2 chance of success
        self.assertEqual(evo.statistics["mut_info_states_signals"], 0)
        self.assertEqual(evo.statistics["avg_prob_success"], 0.5)

        # Run for 10 iterations
        evo.run(10, calculate_stats="end")

        # Check other methods
        # Will still be pooling after only 10 steps
        self.assertTrue(evo.is_pooling())

        # Check is instance
        self.assertIsInstance(evo, evolve.MatchingSR)

    def test_MatchingSRInvention(self):
        """
        Test MatchingSRInvention class

        Returns
        -------
        None.

        """

        # Create random data for standard 2x2 game
        p1_random = np.round(np.random.random(), 4)
        state_chances = np.array([p1_random, 1 - p1_random])
        sender_payoff_matrix = np.eye(2)
        receiver_payoff_matrix = np.eye(2)
        messages = 1

        # Create game
        game = games.Chance(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages=messages,
        )

        # Players start with only 1 "phantom" signal available.
        sender_strategies = np.ones((2, 1))
        receiver_strategies = np.ones((1, 2))

        # Create simulation
        evo = evolve.MatchingSRInvention(game, sender_strategies, receiver_strategies)

        # Check initial stats
        evo.calculate_stats()

        # Should be zero signals to start with
        self.assertEqual(evo.statistics["number_of_signals"], 0)

        # Run for 10 iterations
        evo.run(10, calculate_stats="end")

        # Check other methods
        # Should still be pooling after only 10 steps
        self.assertTrue(evo.is_pooling())

        # Check is instance
        self.assertIsInstance(evo, evolve.MatchingSRInvention)

    def test_MatchingSIR(self):
        """
        Test MatchingSIR class

        Returns
        -------
        None.

        """

        # Set parameters
        p1_random = np.round(np.random.random(), 4)
        state_chances = np.array([p1_random, 1 - p1_random])
        sender_payoff_matrix = np.eye(2)
        intermediary_payoff_matrix = np.eye(2)
        receiver_payoff_matrix = np.eye(2)
        messages_sender = 2
        messages_intermediary = 2

        # Create game
        game = games.ChanceSIR(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            intermediary_payoff_matrix=intermediary_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages_sender=messages_sender,
            messages_intermediary=messages_intermediary,
        )

        # Define initial strategies
        sender_strategies = np.ones((2, 2))
        intermediary_strategies = np.ones((2, 2))
        receiver_strategies = np.ones((2, 2))

        # Create simulation
        evo = evolve.MatchingSIR(
            game, sender_strategies, intermediary_strategies, receiver_strategies
        )

        # Check initial stats
        evo.calculate_stats()

        # Should be 0.5
        self.assertEqual(evo.statistics["prob_success"], 0.5)

        # Run for 10 iterations
        evo.run(10, calculate_stats="end")

        # Since the method should have calculated the stats at the end, check evo.statistics["prob_success"] has length 2
        self.assertEqual(2, len(evo.statistics["prob_success"]))

        # Check is instance
        self.assertIsInstance(evo, evolve.MatchingSIR)

    def test_BushMostellerSR(self):
        """
        Test BushMostellerSR class

        Returns
        -------
        None.

        """

        # Create random data for standard 2x2 game
        p1_random = np.round(np.random.random(), 4)
        state_chances = np.array([p1_random, 1 - p1_random])
        sender_payoff_matrix = np.eye(2)
        receiver_payoff_matrix = np.eye(2)
        messages = 2

        # Create game
        game = games.Chance(
            state_chances=state_chances,
            sender_payoff_matrix=sender_payoff_matrix,
            receiver_payoff_matrix=receiver_payoff_matrix,
            messages=messages,
        )

        # Define strategies
        # These are the initial weights for Bush-Mosteller reinforcement.
        sender_strategies = (
            np.ones((2, 2)) * 0.5
        )  # Strategies are now conditional prob distributions on each row
        receiver_strategies = (
            np.ones((2, 2)) * 0.5
        )  # Strategies are now conditional prob distributions on each row

        # Create a random learning parameter
        learning_param = np.random.randint(1, 50) * 0.01

        # Create simulation object
        evo = evolve.BushMostellerSR(
            game, sender_strategies, receiver_strategies, learning_param
        )

        # Check initial stats
        evo.calculate_stats()

        # Should be zero mutual info and 0.5 probability of success
        self.assertEqual(evo.statistics["mut_info_states_signals"], 0)
        self.assertEqual(evo.statistics["avg_prob_success"], 0.5)

        # Run for 10 iterations
        evo.run(10, calculate_stats="end")

        # Each statistic should have length 2
        self.assertEqual(2, len(evo.statistics["mut_info_states_signals"]))
        self.assertEqual(2, len(evo.statistics["avg_prob_success"]))

        # Check is instance
        self.assertIsInstance(evo, evolve.BushMostellerSR)

    def test_Agent(self):
        """
        Test Agent class

        Returns
        -------
        None.

        """

        # Create deterministic strategies
        strategies = np.eye(3)

        # Create agent objects
        agent = evolve.Agent(strategies)
        sender = evolve.Sender(strategies)
        receiver = evolve.Receiver(strategies)

        # We told the agent to ALWAYS perform act 0 in state 0,
        # act 1 in state 1, and act 2 in state 2.
        # So check it does that.
        self.assertEqual(agent.choose_strategy(0), 0)
        self.assertEqual(agent.choose_strategy(1), 1)
        self.assertEqual(agent.choose_strategy(2), 2)

        # Check the agent can update its strategies
        randit = np.random.randint(0, 3)
        agent.update_strategies(randit, randit, payoff=np.random.random())

        # Check the agent can update its strategies according to the
        # Bush-Mosteller rule
        randit = np.random.randint(0, 3)
        learning_parameter = np.random.randint(1, 50) * 0.01
        agent.update_strategies_bush_mosteller(
            randit,
            randit,
            payoff=np.random.random(),
            learning_parameter=learning_parameter,
        )

        # # Check adding signals works
        # agent.add_signal() # Not implemented
        sender.add_signal()
        receiver.add_signal()

        # Check is instance
        self.assertIsInstance(agent, evolve.Agent)

    def test_Times(self):
        """
        Test Times class

        Returns
        -------
        None.

        """

        # Create evolve.Times object to test replicator methods
        initial_time = 0
        final_time = np.random.randint(5, 20)
        time_inc = 0.05
        times = evolve.Times(initial_time, final_time, time_inc)

        # Check the values of the time vector are as expected
        self.assertEqual(times.time_vector[0], initial_time)
        self.assertEqual(times.time_vector[-1], final_time)

        # Check the time vector is the expected length
        self.assertEqual(
            len(times.time_vector), int((final_time - initial_time) / time_inc) + 1
        )

        # Check is instance
        self.assertIsInstance(times, evolve.Times)


if __name__ == "__main__":
    unittest.main()
