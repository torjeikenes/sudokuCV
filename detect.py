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

    grid = max(cnts,key=cv2.contourArea)

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
    
    blur = cv2.GaussianBlur(img,(7,7),0)
    #blur = cv2.medianBlur(img,5)
    clean = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #blur = cv2.bilateralFilter(img,5,


    # Empty sudoku grid to be filled 
    sudokuMatrix = np.zeros((9,9),dtype=np.uint8)
    out = img.copy()
    # Iterates trough all the cells from top to bottom left to right
    for i,r in enumerate(numberGrid):
        for j,c in enumerate(r):
            logging.debug("row: "+ str(i) + " col: "+ str(j))

            x,y,w,h = cv2.boundingRect(c)
            #cellCrop = blur[y:y+h,x:x+w]
            cellCrop = clean[y+cropMarg:y+h-cropMarg,x+cropMarg:x+w-cropMarg]
            #cell = np.zeros(img.shape,dtype=np.uint8)
            # Single out one cell
            #cell = cv2.drawContours(cell,[c],-1,(255,255,255),-1)
            #cellImg = cv2.bitwise_and(blur,cell)
            number = getNumber(cellCrop)
            sudokuMatrix[i,j] = number
    
    logging.debug("\n"+str(sudokuMatrix))

def getNumber(cellImg):
    # Tresholding before ocr
    #clean = cv2.adaptiveThreshold(cellImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY,5,2)

    

    #dilated = cv2.dilate(clean,None,iterations=1)
    #eroded = cv2.erode(clean,None,iterations=2)

    # Apply OCR on the cropped image 
    config = ('-l eng --oem 1 --psm 10')
    num = pytesseract.image_to_string(cellImg, config=config) 

    if len(num) == 1 and num.isdigit():
        return int(num)
    elif len(num) > 1:
        for c in num:
            if c.isdigit():
                return int(c)
    
    return 0




def main(image):
    #preprocessing
    resize = imutils.resize(image,height=700)
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)

    binary = binaryImage(gray)
    pts = getGridCorners(binary)
    binWarp = perspective.four_point_transform(binary,pts)
    cellMask = filterOutNumber(binWarp)

    grayWarp = perspective.four_point_transform(gray,pts)
    getMatrix(grayWarp,cellMask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect a sudoku grid")
    parser.add_argument("-i","--image",help="Path to image")
    args = vars(parser.parse_args())
    image = cv2.imread(args["image"])
    main(image)
