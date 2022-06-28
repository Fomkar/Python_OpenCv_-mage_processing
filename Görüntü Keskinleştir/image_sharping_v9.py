# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:53:38 2022

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

import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter


original_image = plt.imread('69_inFrame_single_blob_2.bmp').astype('uint16')

# Convert to grayscale
#gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Median filtering


# Calculate the Laplacian
lap = cv2.Laplacian(original_image,cv2.CV_64F)

# Calculate the sharpened image
sharp = original_image - 0.7*lap

cv2.imshow("sharped", sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

