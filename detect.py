#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Detect a sudoku grid
"""

import cv2
import imutils
from imutils import perspective
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -  %(levelname)s -  %(message)s')


def binaryImage(gray):
    """Returns binary image"""

    # Preprocessing
    blur = cv2.GaussianBlur(gray,(11,11),0)

    # Inverse tresholding to later filter contours
    gausTresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,11,2)

    # Dilate and erode to make sure the grid contour connects
    dilated = cv2.dilate(gausTresh,None,iterations=1)
    eroded = cv2.erode(dilated,None,iterations=1)
    

    return eroded

def getGridCorners(binaryImg):
    '''Return conrers'''
    #contours = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    grid = max(contours,key=cv2.contourArea)

    # Find rectangle to get corners
    epsilon = 0.1*cv2.arcLength(grid,True)
    approx = cv2.approxPolyDP(grid,epsilon,True)

    # Convert approx to a 2d array
    pts = approx[:,0,:]

    # Draw corners of grid
    #out = binaryImg.copy()
    #for corner in approx:
    #    cornerTuple = tuple(corner[0])
    #    out = cv2.circle(out,cornerTuple,5,(120,120,120),-1)

    return pts

def filterOutNumber(warp):
    contours = cv2.findContours(warp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    mask = warp.copy()

    # expected size of a cell
    cellVol = warp.shape[0]//9*warp.shape[1]//9
    
    # Filtering out numbers
    for c in contours:
        if cv2.contourArea(c)< cellVol:
            cv2.drawContours(mask,[c],-1,(0,0,0),-1)


    cv2.imshow("CV-warp",mask)
    cv2.waitKey(0)
    
    # Repair grid lines
    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,verticalKernel,iterations=5)
    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,horizontalKernel,iterations=5)

    cv2.imshow("CV-warp",mask)
    cv2.waitKey(0)


    inverted = cv2.bitwise_not(mask)
    






def main(image):
    #preprocessing
    resize = imutils.resize(image,height=500)
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)

    binary = binaryImage(gray)
    pts = getGridCorners(binary)
    warp = perspective.four_point_transform(binary,pts)
    filterOutNumber(warp)
    cv2.imshow("CV-warp",warp)
    cv2.waitKey(0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect a sudoku grid")
    parser.add_argument("-i","--image",help="Path to image")
    args = vars(parser.parse_args())
    image = cv2.imread(args["image"])
    main(image)
