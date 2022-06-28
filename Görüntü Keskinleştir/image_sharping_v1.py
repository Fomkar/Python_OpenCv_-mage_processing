# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:28:08 2022

@author: Gitek_Micro

image keskinleştirme 
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
        cv2.waitKey(0)
        

text = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

text = np.array(text)

print (text[0 ,0])
print (text[1,1])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
cv2.destroyAllWindows()