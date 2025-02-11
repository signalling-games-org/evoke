# Test scripts for evoke.examples.skyrms2010signals module

import unittest
import matplotlib
import warnings
import numpy as np

from evoke.examples.skyrms2010signals import *

# Suppress figure creation
matplotlib.use('Agg')

# Suppress specific matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*FigureCanvasAgg is non-interactive*")

class TestSkrms2010Signals(unittest.TestCase):

    def test_Skyrms2010_1_1(self):
        """
        Test figure class Skyrms2010_1_1 implementing Skyrms (2010) figure 1.1
        """

        # Create the model
        fig = Skyrms2010_1_1()

        # Check values
        self.assertTrue(np.array_equal(fig.state_chances, np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(fig.sender_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(fig.receiver_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertEqual(fig.messages, 2)

        # Check that X, Y, U, V are not None
        self.assertIsNotNone(fig.X)
        self.assertIsNotNone(fig.Y)
        self.assertIsNotNone(fig.U)
        self.assertIsNotNone(fig.V)

    def test_Skyrms2010_1_2(self):
        """
        Test figure class Skyrms2010_1_2 implementing Skyrms (2010) figure 1.2
        """

        # Create the model
        fig = Skyrms2010_1_2()

        # Check values
        self.assertTrue(np.array_equal(fig.payoffs,np.array(
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
        )))

        self.assertTrue(np.array_equal(fig.playertypes, np.array(
            [
                [1, 0, 0, 0],  # "I'm playing the first strategy!"
                [0, 1, 0, 0],  # "I'm playing the second strategy!"
                [0, 0, 1, 0],  # "I'm playing the third strategy!"
                [0, 0, 0, 1],  # "I'm playing the fourth strategy!"
            ]
        )))

        # Check that X, Y, Z, U, V, W are not None
        self.assertIsNotNone(fig.X)
        self.assertIsNotNone(fig.Y)
        self.assertIsNotNone(fig.Z)
        self.assertIsNotNone(fig.U)
        self.assertIsNotNone(fig.V)
        self.assertIsNotNone(fig.W)

    def test_Skyrms2010_3_3(self):
        """
        Test figure class Skyrms2010_3_3 implementing Skyrms (2010) figure 3.3
        """

        # Create the model
        fig = Skyrms2010_3_3()

        # Check values
        self.assertTrue(np.array_equal(fig.state_chances, np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(fig.sender_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(fig.receiver_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertEqual(fig.messages, 2)

        # Check the info statistic exists
        self.assertIsNotNone(fig.evo.statistics["mut_info_states_signals"])

    def test_Skyrms2010_3_4(self):
        """
        Test figure class Skyrms2010_3_4 implementing Skyrms (2010) figure 3.4
        """

        # Create the model
        fig = Skyrms2010_3_4()

        # Check values
        self.assertTrue(np.array_equal(fig.state_chances, np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(fig.sender_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(fig.intermediary_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.array_equal(fig.receiver_payoff_matrix, np.array([[1, 0], [0, 1]])))
        self.assertEqual(fig.messages_sender, 2)
        self.assertEqual(fig.messages_intermediary, 2)

        # Check the probability of success statistic exists
        self.assertIsNotNone(fig.evo.statistics["prob_success"])
