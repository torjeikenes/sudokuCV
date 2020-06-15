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


class TestDetect(unittest.TestCase):
    dir = os.path.dirname(__file__)
    dataPath = os.path.join(dir,'data/')
    numPath = os.path.join(dataPath,'numbers/')
    
    def test_get_numbers(self):
        for i in range(10):
            img = cv2.imread(os.path.join(self.numPath,str(i)+".png"))
            self.assertEqual(detect.getNumber(img),i, "Should be "+str(i))
    
    def test_corners(self):
        img = cv2.imread(os.path.join(self.dataPath,"binary1.png"))
        binIm = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("CV",img)
        #cv2.waitKey(0)
        pts = detect.getGridCorners(binIm)
        self.assertEqual(len(pts),4,"Should have 4 corners")

if __name__ == '__main__':
    unittest.main()
