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
from imutils import contours
import argparse
import logging
import numpy as np
import pytesseract

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
    cnts = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    assert len(cnts)>0, "No contours found"

    grid = max(cnts,key=cv2.contourArea)

    # Find rectangle to get corners
    epsilon = 0.1*cv2.arcLength(grid,True)
    approx = cv2.approxPolyDP(grid,epsilon,True)

    # Convert approx to a 2d array
    pts = approx[:,0,:]

    assert len(pts) == 4, "pts does not have 4 corners"
    return pts

def filterOutNumber(warp):
    cnts = cv2.findContours(warp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    mask = warp.copy()

    # expected size of a cell
    cellVol = warp.shape[0]//9*warp.shape[1]//9
    
    # Filtering out numbers
    for c in cnts:
        if cv2.contourArea(c)< cellVol:
            cv2.drawContours(mask,[c],-1,(0,0,0),-1)

    # Repair grid lines
    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,verticalKernel,iterations=5)
    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,horizontalKernel,iterations=5)

    inverted = cv2.bitwise_not(mask)

    return inverted
    
    
def getMatrix(img,mask):
    '''Return matrix with the sudoku numbers'''
    cropMarg = 2

    cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    assert len(cnts) == 81, "Not 9x9 grid"

    # Sort the contours from top to bottom
    verticalSort,_ = contours.sort_contours(cnts,method="top-to-bottom")

    numberGrid = []
    row = []
    # Sorts the cells on the row from left to right
    for i,c in enumerate(verticalSort,1):
        row.append(c)
        if i%9 == 0:
            horizontalSort,_ = contours.sort_contours(row,method="left-to-right")
            numberGrid.append(horizontalSort)
            row = []
    
    # Preprocessing
    blur = cv2.GaussianBlur(img,(5,5),0)
    clean = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    # Empty sudoku grid to be filled 
    sudokuMatrix = np.zeros((9,9),dtype=np.uint8)

    # Iterates trough all the cells from top to bottom left to right
    for i,r in enumerate(numberGrid):
        for j,c in enumerate(r):
            # Crops the cell
            x,y,w,h = cv2.boundingRect(c)
            cellCrop = clean[y+cropMarg:y+h-cropMarg,x+cropMarg:x+w-cropMarg]
            # Get the number in the cell and add it to the matrix
            number = getNumber(cellCrop)
            logging.debug("{}, {}: {}".format(i,j,number))
            sudokuMatrix[i,j] = number
    
    return sudokuMatrix

def getNumber(cellImg):
    '''Returns the number from a cell'''
    # TODO: increase speed of OCR
    # Apply OCR on the cropped image 
    config = ('-l eng --oem 1 --psm 10')
    num = pytesseract.image_to_string(cellImg, config=config) 

    # Only returns a single digit
    if len(num) == 1 and num.isdigit():
        return int(num)
    elif len(num) > 1:
        for c in num:
            if c.isdigit():
                return int(c)
    return 0

def img2Matrix(image):
    '''Returns a sudoku matrix given an image'''
    #preprocessing
    resize = imutils.resize(image,height=700)
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    binary = binaryImage(gray)
        
    # Transforms and crops mask and grayscale img to sudoku grid
    pts = getGridCorners(binary)
    binWarp = perspective.four_point_transform(binary,pts)
    grayWarp = perspective.four_point_transform(gray,pts)

    # Clean up to only grid
    cellMask = filterOutNumber(binWarp)

    #Get sudoku matrix
    matrix = getMatrix(grayWarp,cellMask)

    return matrix

def main(image):
    try:
        matrix = img2Matrix(image)
    except AssertionError as err:
        print("img2Matrix Failed: {}".format(err))

    logging.info("\n"+str(matrix))
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect a sudoku grid")
    parser.add_argument("-i","--image",help="Path to image")
    args = vars(parser.parse_args())
    image = cv2.imread(args["image"])
    main(image)
