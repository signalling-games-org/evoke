# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:14:59 2024

@author: stephenfmann

Unit tests for evoke/examples/skyrms2010signals.py
"""


import unittest
import matplotlib

import evoke.examples.skyrms2010signals as skyrms


class TestFigure(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        # Suppress figure window pop-ups
        matplotlib.use("Agg")

    def test_Skyrms2010_1_1(self):
        """
        Test Skyrms2010_1_1 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_1_1()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_1_1)

    def test_Skyrms2010_1_2(self):
        """
        Test Skyrms2010_1_2 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_1_2()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_1_2)

    def test_Skyrms2010_3_3(self):
        """
        Test Skyrms2010_3_3 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_3_3()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_3_3)

    def test_Skyrms2010_3_4(self):
        """
        Test Skyrms2010_3_4 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_3_4()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_3_4)

    def test_Skyrms2010_4_1(self):
        """
        Test Skyrms2010_4_1 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_4_1()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_4_1)

    def test_Skyrms2010_5_2(self):
        """
        Test Skyrms2010_5_2 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_5_2()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_5_2)

    def test_Skyrms2010_8_1(self):
        """
        Test Skyrms2010_8_1 class

        Returns
        -------
        None.

        """

        # Create object
        f = skyrms.Skyrms2010_8_1()

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_8_1)

    def test_Skyrms2010_8_2(self):
        """
        Test Skyrms2010_8_2 class

        Returns
        -------
        None.

        """

        # Create object
        # Use small parameters for testing
        f = skyrms.Skyrms2010_8_2(10, 100)

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_8_2)

    def test_Skyrms2010_8_3(self):
        """
        Test Skyrms2010_8_3 class

        Returns
        -------
        None.

        """

        # Create object
        # Use small parameters for testing
        f = skyrms.Skyrms2010_8_3(10, 500)

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_8_3)

    def test_Skyrms2010_10_5(self):
        """
        Test Skyrms2010_10_5 class

        Returns
        -------
        None.

        """

        # Create object
        # Use small parameters for testing
        f = skyrms.Skyrms2010_10_5(50, 500)

        # Check it is of the correct type
        self.assertIsInstance(f, skyrms.Skyrms2010_10_5)


if __name__ == "__main__":
    unittest.main()
