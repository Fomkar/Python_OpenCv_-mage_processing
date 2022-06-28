# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:13:14 2022

@author: Gitek_Micro
"""
# Canny Edge DEtection
import numpy as np
import cv2
image = cv2.imread('elma_kesik2022-03-1112-55-49_3.jpg',0)
cv2.imshow("original_image",image)
#cv2.waitKey(0)
def nothing(x):
    pass




cv2.namedWindow('image')
cv2.createTrackbar('trackbar','image',0,400,nothing)
cv2.createTrackbar('trackbar_2','image',0,400,nothing)
while True:
     r = cv2.getTrackbarPos('trackbar','image')
     b = cv2.getTrackbarPos('trackbar','image')
     edges = cv2.Canny(image,b,r)
     cv2.imshow('image', edges)
     k = cv2.waitKey(1) & 0xFF
     if k == 27:
         break
#cv2.imshow("edges", edges)
#cv2.waitKey(0)

cv2.destroyAllWindows()
































