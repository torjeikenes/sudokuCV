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
    '''solve sudoku grid. Called recursively'''
    solvedGrid = None
    for i in range(9):
        for j in range(9):
            if grid[i,j] == 0:
                for n in range(1,10):
                    if checkNum(grid,n,i,j):
                        grid[i,j] = n
                        solvedGrid, solved = solve(grid)
                        if solved:
                            return solvedGrid, solved
                        grid[i,j] = 0
                return grid,False
    #logging.info("solved: "+str(grid))
    return grid, True

def checkNum(grid,n,y,x):
    '''Check if the number is possible'''
    possible = True
    # Check column
    if n in grid[y]:
        possible = False
    # Check row
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

