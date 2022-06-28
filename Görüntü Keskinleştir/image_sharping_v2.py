# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:30:19 2022

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
        image = cv2.imread(f,0)
        print(f)
        cv2.imshow(("Original image"+ str(i)), image)

        kernel = np.array([[ 0, -1, 0],
                           [-1, 5, -1],
                           [ 0, -1, 0]])
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        cv2.imshow('AV CV- Winter Wonder Sharpened', image_sharp)
        cv2.waitKey(0)
cv2.destroyAllWindows()