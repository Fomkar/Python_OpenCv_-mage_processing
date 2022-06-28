# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:33:36 2022

@author: Gitek_Micro
"""

import numpy as np
import sys
import cv2 

#Sobel Edge detection


image = cv2.imread("elma_kesik2022-03-1112-55-49_3.jpg", cv2.IMREAD_COLOR)
cv2.imshow("original_image",image)
cv2.waitKey(0)

if image is None:
    print ('Error opening image: ')
 

    
image = cv2.GaussianBlur(image, (3, 3), 0)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray_image",gray)
cv2.waitKey(0)

grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1.9, delta=0, borderType=cv2.BORDER_DEFAULT)
# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1.9, delta=0, borderType=cv2.BORDER_DEFAULT)


abs_grad_x = cv2.convertScaleAbs(grad_x)
cv2.imshow("Sobel filtre_x", abs_grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
cv2.imshow("Sobel filtre_y", abs_grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


cv2.imshow("Sobel filtre", grad)
cv2.waitKey(0)

cv2.destroyAllWindows()
