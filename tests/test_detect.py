#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 torje <torje@torje-ubuntu>
#
# Distributed under terms of the MIT license.

"""
Testing of detect module
"""

import unittest
import os
from .context import detect
import cv2
import numpy as np
import imutils

def is_similar(image1, image2):
    '''Check if 2 images are similar'''
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

class TestDetect(unittest.TestCase):
    dir = os.path.dirname(__file__)
    dataPath = os.path.join(dir,'data/')
    numPath = os.path.join(dataPath,'numbers/')
    correctMtrx = np.array([[0, 0, 0, 0, 5, 0, 0, 7, 0],
                            [0, 0, 0, 3, 0, 0, 2, 5, 0],
                            [0, 0, 0, 0, 0, 4, 0, 3, 8],
                            [0, 0, 0, 0, 7, 6, 4, 0, 3],
                            [1, 0, 0, 0, 0, 0, 0, 0, 2],
                            [9, 0, 3, 2, 8, 0, 0, 0, 0],
                            [4, 5, 0, 1, 0, 0, 0, 0, 0],
                            [0, 8, 6, 0, 0, 5, 0, 0, 0],
                            [0, 7, 0, 0, 9, 0, 0, 0, 0]])

    def test_binary(self):
        img = cv2.imread(os.path.join(self.dataPath,"sudoku1.jpg"))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        binary = detect.binaryImage(gray)
        self.assertEqual(len(binary.shape),2,
                        "Binary should be 2D. It has "+str(len(binary.shape)))
        self.assertEqual(binary.shape,gray.shape,
                        "Size should not change. size is "+ str(binary.shape))

    def test_corners(self):
        binIm = cv2.imread(os.path.join(self.dataPath,"binary1.png"),
                         cv2.IMREAD_GRAYSCALE)
        pts = detect.getGridCorners(binIm)
        self.assertEqual(len(pts),4,"Should have 4 corners. It has "+ str(len(pts)))
    
    def test_get_numbers(self):
        for i in range(10):
            img = cv2.imread(os.path.join(self.numPath,str(i)+".png"))
            num = detect.getNumber(img)
            self.assertEqual(num,i, 
                            "Should be {}. {} was detected".format(i,num))
    def test_filter(self):
        warp = cv2.imread(os.path.join(self.dataPath,"binaryWarp.png"),
                          cv2.IMREAD_GRAYSCALE)
        filtered = detect.filterOutNumber(warp);
        cells = cv2.findContours(filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cells = imutils.grab_contours(cells)
        self.assertEqual(len(cells),81,"Should have 81 cells. Has "+ str(len(cells)))

    def test_matrix(self):
        warp = cv2.imread(os.path.join(self.dataPath,"grayWarp.png"),
                          cv2.IMREAD_GRAYSCALE)
        cellMask = cv2.imread(os.path.join(self.dataPath,"cellMask.png"),
                          cv2.IMREAD_GRAYSCALE)
        matrix,cellPos = detect.getMatrix(warp,cellMask) 
        self.assertTrue(np.array_equal(matrix,self.correctMtrx),
              "Matrix detection failed. Detected {}\n correct{}".format(str(matrix), str(self.correctMtrx)))
    
    def test_drawPoint(self):
        img = cv2.imread(os.path.join(self.dataPath,"resize.png"))
        points = cv2.imread(os.path.join(self.dataPath,"points.png"))
        pts = np.array([[160,183],
                        [577,101],
                        [286,587],
                        [771,451]])
        drawnPoints = detect.drawPoints(img,pts)
        self.assertTrue(is_similar(points,drawnPoints),"Points image is not correct")
        
    def test_img2matrix(self):
        img = cv2.imread(os.path.join(self.dataPath,"sudoku1.jpg"))
        matrix,_,_,_,_ = detect.img2Matrix(img) 
        self.assertEqual(matrix.tolist(),self.correctMtrx.tolist(),
              "Matrix detection failed. Detected {}\n correct{}".format(str(matrix), str(self.correctMtrx)))
    
    def test_detect_main(self):
        img = cv2.imread(os.path.join(self.dataPath,"sudoku1.jpg"))
        matrix = detect.main(img)
        self.assertEqual(matrix.tolist(),self.correctMtrx.tolist(),
              "Matrix detection failed. Detected {}\n correct{}".format(str(matrix), str(self.correctMtrx)))

        
if __name__ == '__main__':
    unittest.main()
