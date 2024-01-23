# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:22:40 2024

@author: stephe
nfmann
Unit tests for evoke/src/figure.py
"""

import unittest
import numpy as np

import evoke.src.figure as figure


class TestFigure(unittest.TestCase):
    def test_scatter(self):
        
        # Create figure object
        f = figure.Scatter()
        
        # Load figure with random data
        f.reset(x=range(10),
                y=np.random.random((10,)),
                xlabel = "this is the x-axis",
                ylabel = "this is the y-axis")
        
        # Check the figure works
        f.show()
        
        # Check the figure is an instance of the appropriate class
        self.assertIsInstance(f, figure.Scatter)


if __name__ == "__main__":
    unittest.main()