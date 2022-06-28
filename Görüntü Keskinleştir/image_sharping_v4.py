# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:40:28 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time


# Görüntüleri okuma ve gösterme 
currentDir = os.getcwd()
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        image = cv2.imread(f)
        cv2.imshow("original", image)
        #kenar bulma 
        kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, 0]], np.float32) 

        kernel = 1/3 * kernel

        sharpened = cv2.filter2D(image, -1, kernel)
        cv2.imshow('Image Sharpening', sharpened)
        cv2.waitKey(0)
cv2.destroyAllWindows()