#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Display solved sudoku on to a image
"""

import cv2
import imutils
from imutils import perspective
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s -  %(message)s')

def display(img,cellPts,sudoku,corners,warpShape):
    textOut = np.zeros((warpShape[0],warpShape[1],3),dtype=np.uint8)
    for i,row in enumerate(sudoku):
        for j,n in enumerate(row):
            textOut = cv2.putText(textOut,str(n),cellPts[i*9+j],# index from 2d to 1d array
                        cv2.FONT_ITALIC,0.9,(255,255,0),2)
    sh = textOut.shape 

    # Orders corners to top-left,top-right,bottom-right, bottom-left
    corners = perspective.order_points(np.array(corners))
    # Ordered to top-left,top-right,bottom-right, bottom-left
    pts = np.float32([[0,0],[sh[1],0],[sh[1],sh[0]],[0,sh[0]]])

    # Transforms image to overlay sudoku on original image
    M = cv2.getPerspectiveTransform(pts,corners)
    warp = cv2.warpPerspective(textOut,M,(img.shape[1],img.shape[0]))

    inv = (255-warp)
    

    out = cv2.bitwise_and(img,inv)
    return out




