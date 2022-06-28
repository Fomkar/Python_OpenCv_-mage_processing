# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:15:26 2022

@author: Gitek_Micro
"""

import cv2 as cv
import numpy as np
import os 
currentDir = "D://Opencv Alıştırmalar/Python/Corner Detection/red_elma"

os.chdir(currentDir)
files = os.listdir()
for f in files:
    if f.endswith(".jpg"):
        # Take each frame
        frame = cv.imread(f)
        # Convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_red = np.array([5,33,0])
        upper_red = np.array([179,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_red, upper_red)
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame,frame, mask= mask)
        cv.namedWindow("frame",cv.WINDOW_NORMAL)
        cv.namedWindow("mask",cv.WINDOW_NORMAL)
        cv.namedWindow("res",cv.WINDOW_NORMAL)
        cv.imshow('frame',frame)
        cv.imshow('mask',mask)
        cv.imshow('res',res)
        cv.waitKey(0)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
cv.destroyAllWindows()
