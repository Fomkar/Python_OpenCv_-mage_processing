# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:59:58 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
import math
#Hocanın Kullandığı daire bulma
img = cv2.imread('elma_kesik2022-03-1112-55-49_3.jpg',0)


#img = cv2.medianBlur(img,1)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,
                            param1=50,param2=12,minRadius=0,maxRadius=20)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)

#Benim kullandığım

image = cv2.imread('elma_kesik2022-03-1112-55-49_3.jpg',1)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_gauss = cv2.GaussianBlur(image_gray, (5, 5), 0)
ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
# get contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(thresh, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
contours_area = []
# calculate area and filter into new array
for con in contours:
    area = cv2.contourArea(con)
    if 1000 < area < 10000:
        contours_area.append(con)


contours_cirles = []

# check if contour is of circular shape
for con in contours_area:
    perimeter = cv2.arcLength(con, True)
    area = cv2.contourArea(con)
    print(area)
    if perimeter == 0:
        break
    circularity = 4*math.pi*(area/(perimeter*perimeter))
    print (circularity)
    if 0.7 < circularity < 1.2:
        contours_cirles.append(con)
        
cv2.imshow("result", image)
cv2.waitKey(0)














cv2.destroyAllWindows()