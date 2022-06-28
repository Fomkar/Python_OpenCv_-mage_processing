# -*- coding: utf-8 -*-
"""
edges.py: Canny, Prewitt and Sobel Edge detection using opencv
"""
__author__      = "Ömer Karagöz"
__email__ = "karagozo240@gmail.com"

import cv2
import numpy as np

image = cv2.imread('elma_kesik2022-03-1112-55-49_3.jpg')
cv2.imshow("original image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow("gaus image", img_gaussian)
#canny
img_canny = cv2.Canny(image,100,200)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)


cv2.imshow("Original Image", image)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()