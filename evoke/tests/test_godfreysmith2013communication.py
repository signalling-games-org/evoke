# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:35:46 2024

@author: stephenfmann

Unit tests for evoke/examples/godfreysmith2013communication.py

NB there is not yet a demo mode for GodfreySmith2013_3 so we cannot test it here.
"""

import unittest

import evoke.examples.godfreysmith2013communication as pgs


class TestFigure(unittest.TestCase):
    def test_GodfreySmith2013_1(self):
        """
        Test GodfreySmith2013_1 class

        Returns
        -------
        None.

        """

        # Create figure in demo mode
        # Just use 10 games per c for quick testing
        f = pgs.GodfreySmith2013_1(games_per_c=10, demo=True)

        # Check figure is of the correct type
        self.assertIsInstance(f, pgs.GodfreySmith2013_1)

    def test_GodfreySmith2013_2(self):
        """
        Test GodfreySmith2013_2 class

        Returns
        -------
        None.

        """

        # Create figure in demo mode
        # Just use 10 games per c for quick testing
        f = pgs.GodfreySmith2013_2(games_per_c=10, demo=True)

        # Check figure is of the correct type
        self.assertIsInstance(f, pgs.GodfreySmith2013_2)


if __name__ == "__main__":
    unittest.main()
