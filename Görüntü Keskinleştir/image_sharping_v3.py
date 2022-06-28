# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:37:00 2022

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
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
        cv2.imshow('Image Sharpening', sharpened)
        cv2.waitKey(0)
cv2.destroyAllWindows()