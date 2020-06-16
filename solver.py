#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Sudoku Solver
"""

import numpy as np

def solve(grid):
    '''solve sudoku grid. Called recursively'''
    for i in range(9):
        for j in range(9)
            if grid[i,j] == 0;
                for n in range(1,10):
                    if checkNum(n,i,j):
                        grid[i,j] = n
                        grid = solve(grid)
                        grid[i,j] = 0
                return grid
    return grid




if __name__ == '__main__':
    grid =np.array([[0, 0, 0, 0, 5, 0, 0, 7, 0],
                    [0, 0, 0, 3, 0, 0, 2, 5, 0],
                    [0, 0, 0, 0, 0, 4, 0, 3, 8],
                    [0, 0, 0, 0, 7, 6, 4, 0, 3],
                    [1, 0, 0, 0, 0, 0, 0, 0, 2],
                    [9, 0, 3, 2, 8, 0, 0, 0, 0],
                    [4, 5, 0, 1, 0, 0, 0, 0, 0],
                    [0, 8, 6, 0, 0, 5, 0, 0, 0],
                    [0, 7, 0, 0, 9, 0, 0, 0, 0]])

