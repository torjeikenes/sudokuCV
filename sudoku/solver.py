#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Sudoku Solver
"""

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s -  %(message)s')

def solve(grid):
    '''solve sudoku grid. Called recursively with backtracking'''
    solvedGrid = None
    assert len(grid) == 9, "Sudoku not 9x9"
    assert len(grid[0]) == 9, "Sudoku not 9x9"
    # Check all cells
    for i in range(9):
        for j in range(9):
            if grid[i,j] == 0:
                for n in range(1,10):
                    if checkNum(grid,n,i,j):# Checks if number is possible in location
                        grid[i,j] = n
                        solvedGrid, solved = solve(grid)
                        if solved: # Should not continue backtracking
                            return solvedGrid, solved
                        # Last attemt did not work, so cell is reset
                        grid[i,j] = 0 
                # Attemt didn't succeed so it backtracks
                return grid, False

    assert 0 not in grid, "Sudoku not solved"
    return grid, True # Sudoku is solved

def checkNum(grid,n,y,x):
    '''Check if the number is possible'''
    possible = True
    # Check row
    if n in grid[y]:
        possible = False
    # Check column
    for i in range(9):
        if grid[i,x] == n:
            possible = False
    # Check square
    squareY = (y // 3)*3
    squareX = (x // 3)*3
    for i in range(squareY,squareY+3):
        for j in range(squareX,squareX+3):
            if grid[i,j] == n:
                possible = False

    return possible


