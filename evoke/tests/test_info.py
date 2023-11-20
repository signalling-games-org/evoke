# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:21:11 2023

@author: ste_m

Unit tests for evoke/lib/info.py
"""

import unittest
import numpy as np

import evoke.lib.info as info


class TestInfo(unittest.TestCase):
    
    def test_conditional_entropy(self):
        
        conds = np.array([[1/2,1/4,1/4], [1/3,1/3,1/3], [0,1/10,9/10]])
        unconds = np.array([1/12, 5/12, 1/2])
        
        self.assertAlmostEqual(
            info.conditional_entropy(conds, unconds), 
            1.019899, 
            places=6)
    
    def test_mutual_info_from_joint(self):
        
        matrix = np.array([[3/20,5/20,0], [3/20,2/20,2/20], [0,1/20,4/20]])
        
        self.assertAlmostEqual(
            info.mutual_info_from_joint(matrix), 
            0.463865, 
            places = 6)
    
    def test_unconditional_probabilities(self):
        # This function does not appear to be used
        pass
    
    def test_normalize_axis(self):
        
        array = np.array([[1,1,1],[2,2,2],[3,3,3]])
        
        # Vertical axis
        axis = 0
        np.testing.assert_array_almost_equal(
            info.normalize_axis(array, axis),
            np.array([[0.1667,0.1667,0.1667],[0.3333,0.3333,0.3333],[0.5,0.5,0.5]]),
            decimal = 4
            )
        
        # Horizontal axis
        axis = 1
        np.testing.assert_array_almost_equal(
            info.normalize_axis(array, axis),
            np.ones((3,3))/3,
            decimal = 4
            )
    

if __name__ == "__main__":
    unittest.main()