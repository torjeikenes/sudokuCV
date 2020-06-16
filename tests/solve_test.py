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

def diff(m1,m2):
    dif = np.abs(m1-m2)
    return str(dif)

class TestSolver(unittest.TestCase):
    grid =np.array([[0,0,0,0,5,0,0,7,0],
                    [0,0,0,3,0,0,2,5,0],
                    [0,0,0,0,0,4,0,3,8],
                    [0,0,0,0,7,6,4,0,3],
                    [1,0,0,0,0,0,0,0,2],
                    [9,0,3,2,8,0,0,0,0],
                    [4,5,0,1,0,0,0,0,0],
                    [0,8,6,0,0,5,0,0,0],
                    [0,7,0,0,9,0,0,0,0]])
    correct =np.array([[6,3,4,8,5,2,1,7,9],
                        [7,1,8,3,6,9,2,5,4],
                        [5,9,2,7,1,4,6,3,8],
                        [8,2,5,9,7,6,4,1,3],
                        [1,6,7,5,4,3,9,8,2],
                        [9,4,3,2,8,1,5,6,7],
                        [4,5,9,1,3,7,8,2,6],
                        [3,8,6,4,2,5,7,9,1],
                        [2,7,1,6,9,8,3,4,5]])
    
    def test_fail_row(self):
        possible = solver.checkNum(self.grid,5,1,4)
        self.assertFalse(possible,"Should not be possible")

    def test_fail_column(self):
        possible = solver.checkNum(self.grid,4,3,2)
        self.assertFalse(possible,"Should not be possible")

    def test_fail_square(self):
        possible = solver.checkNum(self.grid,9,7,4)
        self.assertFalse(possible,"Should not be possible")

    def test_possible(self):
        possible = solver.checkNum(self.grid,5,4,3)
        self.assertTrue(possible,"Should be possible")

    def test_solve(self):
        solved = self.grid.copy()
        solved,_ = solver.solve(solved)
        self.assertTrue(np.array_equal(solved,self.correct),
                        "Not solved correctly. solved: \n"+ str(solved) + "Diff"+ str(diff(solved,self.correct)))

    def test_solve_impossible(self):
        solved = self.grid.copy()
        solved[1,0] = 6
        solved,completed = solver.solve(solved)
        self.assertFalse(completed, "Should not be solvable")

if __name__ == '__main__':
    unittest.main()
