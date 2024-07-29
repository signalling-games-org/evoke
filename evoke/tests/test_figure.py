# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:22:40 2024

@author: stephe
nfmann
Unit tests for evoke/src/figure.py
"""

import unittest
import numpy as np
import matplotlib

import evoke.src.figure as figure


class TestFigure(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):

        # Suppress figure window pop-ups
        matplotlib.use("Agg")
    
    def test_scatter(self):
        """
        Test figure.Scatter class

        Returns
        -------
        None.

        """

        # Create figure object
        f = figure.Scatter()

        # Create random data
        x_data = range(10)
        y_data = np.random.random((10,))

        # Load figure with random data
        f.reset(
            x=x_data,
            y=y_data,
            xlabel="this is the x-axis",
            ylabel="this is the y-axis",
        )

        # Check the figure works
        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Scatter)

    def test_quiver_2D(self):
        """
        Test figure.Quiver2D class


        """

        # Create figure object
        f = figure.Quiver2D()

        # Load figure with random data
        freqs = np.linspace(0.01, 0.99, 15)
        f.X, f.Y = np.meshgrid(freqs, freqs)

        def uv_from_xy(x, y):
            # Fake data
            l = np.random.random((2,)).tolist()

            return l[0], l[1]

        f.U, f.V = np.vectorize(uv_from_xy)(f.X, f.Y)

        # Check the figure works
        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Quiver2D)

    def test_quiver_3D(self):
        """
        Test figure.Quiver3D class

        Returns
        -------
        None.

        """

        # Create figure object
        f = figure.Quiver3D()

        # Load figure with random data TODO randomise
        pop_vectors = np.random.random((8, 3)) * 2

        # Random arrow directions
        arrows = np.random.random((8, 3))

        # Prep data
        soa = np.hstack((pop_vectors, arrows))

        # Give data to class in nice format
        f.X, f.Y, f.Z, f.U, f.V, f.W = zip(*soa)

        # Check the figure works
        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Quiver3D)

    def test_bar(self):
        """
        Test figure.Bar class

        Returns
        -------
        None.

        """

        # Create figure object
        f = figure.Bar()

        # Create random data
        x = range(np.random.randint(100))
        y = np.random.random((len(x),))
        xlabel = "Random data for test x-axis"
        ylabel = "Random data for test y-axis"

        # Load data into figure
        f.reset(x, y, xlabel, ylabel)

        # Show figure
        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Bar)

    def test_ternary(self):
        """
        Test figure.Ternary class

        Returns
        -------
        None.

        """

        # Create figure object
        f = figure.Ternary()

        # Create random data
        # "xyzs is a list of arrays of dimensions nx3, such that each row is a
        # 3-dimensional stochastic vector.  That is to say, for now, a collection
        # of orbits"
        # Let's just do 1 orbit
        f.xyzs = [np.random.random((10, 3))]

        f.reset(
            right_corner_label="Test right corner",
            top_corner_label="Test top corner",
            left_corner_label="Test left corner",
            fontsize=10,
        )

        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Ternary)

    def test_surface(self):
        """
        Test figure.Surface class

        Returns
        -------
        None.

        """

        # Create figure objet
        f = figure.Surface()

        # Create random data
        lim = np.random.randint(100)
        x = range(lim)
        y = range(lim)
        z = np.random.random((lim, lim))

        # Load data into figure
        f.reset(x=x, y=y, z=z)

        # Show figure
        f.show()

        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Surface)


if __name__ == "__main__":
    unittest.main()
