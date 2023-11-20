# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:21:11 2023

@author: ste_m

Unit tests for evoke/lib/info.py
"""

import unittest
import numpy as np

import evoke.src.info as info


class TestInfo(unittest.TestCase):
    def test_conditional_entropy(self):
        conds = np.array(
            [[1 / 2, 1 / 4, 1 / 4], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 10, 9 / 10]]
        )
        unconds = np.array([1 / 12, 5 / 12, 1 / 2])

        self.assertAlmostEqual(
            info.conditional_entropy(conds, unconds), 1.019899, places=6
        )

    def test_mutual_info_from_joint(self):
        matrix = np.array(
            [[3 / 20, 5 / 20, 0], [3 / 20, 2 / 20, 2 / 20], [0, 1 / 20, 4 / 20]]
        )

        self.assertAlmostEqual(info.mutual_info_from_joint(matrix), 0.463865, places=6)

    def test_unconditional_probabilities(self):
        # This function does not appear to be used
        pass

    def test_normalize_axis(self):
        array = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        # Vertical axis
        axis = 0
        np.testing.assert_array_almost_equal(
            info.normalize_axis(array, axis),
            np.array(
                [[0.1667, 0.1667, 0.1667], [0.3333, 0.3333, 0.3333], [0.5, 0.5, 0.5]]
            ),
            decimal=4,
        )

        # Horizontal axis
        axis = 1
        np.testing.assert_array_almost_equal(
            info.normalize_axis(array, axis), np.ones((3, 3)) / 3, decimal=4
        )

    def test_from_conditional_to_joint(self):
        # Define unconditional probabilities
        unconds = np.array([1 / 12, 5 / 12, 1 / 2])

        # Define conditional probabilities
        conds = np.array(
            [[1 / 2, 1 / 4, 1 / 4], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 10, 9 / 10]]
        )

        # Define joint probabilities
        joint = np.array(
            [[0.0417, 0.0208, 0.0208], [0.1389, 0.1389, 0.1389], [0.0, 0.05, 0.45]]
        )

        np.testing.assert_array_almost_equal(
            info.from_conditional_to_joint(unconds, conds), joint, decimal=4
        )

    def test_bayes_theorem(self):
        # Define unconditional probabilities
        unconds = np.array([1 / 12, 5 / 12, 1 / 2])

        # Define conditional probabilities
        conds = np.array(
            [[1 / 2, 1 / 4, 1 / 4], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 10, 9 / 10]]
        )

        # Define what the result should be
        result = np.array(
            [[0.2308, 0.0993, 0.0342], [0.7692, 0.6623, 0.2278], [0.0, 0.2384, 0.7380]]
        )

        np.testing.assert_array_almost_equal(
            info.bayes_theorem(unconds, conds), result, decimal=4
        )

    def test_entropy(self):
        # Define vector
        vector = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

        self.assertEqual(info.entropy(vector), 2)

    def test_escalar_product_map(self):
        """
        Take a matrix and a vector and return a matrix consisting of each element
        of the vector multiplied by the corresponding row of the matrix
        """

        matrix = np.array(
            [[1 / 2, 1 / 4, 1 / 4], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 10, 9 / 10]]
        )
        vector = np.array([1 / 12, 5 / 12, 1 / 2])

        result = np.array(
            [[0.0417, 0.0208, 0.0208], [0.1389, 0.13889, 0.1389], [0.0, 0.05, 0.45]]
        )

        np.testing.assert_array_almost_equal(
            info.escalar_product_map(matrix, vector), result, decimal=4
        )

    def test_normalize_vector(self):
        vector = np.array([1, 2, 3, 4, 5])

        result = np.array([1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15])

        np.testing.assert_array_almost_equal(
            info.normalize_vector(vector), result, decimal=4
        )

    def test_normalize_distortion(self):
        """
        Normalize linearly so that max corresponds to 0 distortion, and min to 1 distortion
        It must be a matrix of floats!
        """

        matrix = np.array(
            [[1 / 2, 1 / 4, 1 / 4], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 10, 9 / 10]]
        )

        result = np.array(
            [[0.4444, 0.7222, 0.7222], [0.6296, 0.6296, 0.6296], [1.0, 0.8889, 0.0]]
        )

        np.testing.assert_array_almost_equal(
            info.normalize_distortion(matrix), result, decimal=4
        )


if __name__ == "__main__":
    unittest.main()
