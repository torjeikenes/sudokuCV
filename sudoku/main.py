#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import detect
import solver
import display
import logging
import argparse
import cv2

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s -  %(message)s')

def main(image):
    try:
        matrix, cellPts, pts, resize,warpSize = detect.img2Matrix(image)
        logging.info("Detected sudoku: \n"+str(matrix))
    except AssertionError as err:
        print("Sudoku detector failed: {}".format(err))
        return

    try:
        solved = solver.solve(matrix)
        logging.info("Solved sudoku: \n"+str(matrix))
    except AssertionError as err:
        print("Sudoku solve failed: {}".format(err))
        return
    
    output = display.display(resize,cellPts,matrix,pts,warpSize)
    cv2.imshow("CV-Sudoku",output)
    cv2.waitKey(0)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve a sudoku from a picture")
    parser.add_argument("-i","--image",help="Path to image")
    args = vars(parser.parse_args())
    image = cv2.imread(args["image"])

    main(image)