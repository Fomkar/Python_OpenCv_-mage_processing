# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:53:29 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time


# Görüntüleri okuma ve gösterme 
#currentDir = "calısmak istediğin dosya yolu"
currentDir = os.getcwd() # .py dosyasının bulunduğu klasör.
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        #print(f)
        image = cv2.imread(f)
        #cv2.imshow("original", image)
        smoothed = cv2.GaussianBlur(image, (9, 9), 10)
        #cv2.imshow("gaus", smoothed)
        unsharped = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
        cv2.imshow("unsharped", unsharped)
        cv2.imwrite(""+f, unsharped)
        cv2.waitKey(5)
cv2.destroyAllWindows()