#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Testing of solver module
"""

import unittest
import os
from .context import solver
import numpy as np


class TestSolver(unittest.TestCase):
    def test_solve(self):
        grid =np.array([[0,0,0,0,5,0,0,7,0],
                        [0,0,0,3,0,0,2,5,0],
                        [0,0,0,0,0,4,0,3,8],
                        [0,0,0,0,7,6,4,0,3],
                        [1,0,0,0,0,0,0,0,2],
                        [9,0,3,2,8,0,0,0,0],
                        [4,5,0,1,0,0,0,0,0],
                        [0,8,6,0,0,5,0,0,0],
                        [0,7,0,0,9,0,0,0,0]])
        solved = solver.solve(grid)
        correct =np.array([[6,3,4,8,5,2,1,7,9],
                           [7,1,8,3,6,9,2,5,4],
                           [5,9,2,7,1,4,6,3,8],
                           [8,2,5,9,7,6,4,1,3],
                           [1,6,7,5,4,3,9,8,2],
                           [9,4,3,2,8,1,5,6,7],
                           [4,5,9,1,3,7,8,2,6],
                           [4,5,9,1,3,7,8,2,6],
                           [2,7,1,6,9,8,3,4,5]])
        self.assertTrue(np.array_equal(solved,correct),"Not solved correctly")


       
if __name__ == '__main__':
    unittest.main()
