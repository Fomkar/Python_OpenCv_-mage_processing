# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:16:25 2022

@author: Gitek_Micro
"""
import cv2
import numpy as np
import os



# Görüntüleri okuma ve gösterme
currentDir = 'C:/Users/Gitek_Micro/Downloads/son eklenenler'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx=0
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        idx +=1 
        print(f)
        img = cv2.imread(f)
        cv2.imshow('Original', img)
        # generating the kernels
        kernel_sharpen_1 = np.array([[-1,-1,-1,-1,-1],
                                     [-1,2,2,2,-1],
                                     [-1,2,8,2,-1],
                                     [-1,2,2,2,-1],
                                     [-1,-1,-1,-1,-1]]) / 8.0
        # applying different kernels to the input image

        output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
        cv2.imshow('Sharpening', output_1)
        #cv2.imwrite("C:/Users/Gitek_Micro/Downloads/Son eklenenler keskin/double_blob_"+ str(idx) +".bmp" , output_1)
        cv2.waitKey(500)
cv2.destroyAllWindows()
