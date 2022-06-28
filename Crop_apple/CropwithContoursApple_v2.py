# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:10:57 2022

@author: Gitek_Micro
"""

import numpy as np
import cv2
image = cv2.imread('elma_kesik2022-03-1112-55-49_3.jpg',1)
cv2.imshow("original_image",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#orijinal görüntü
cv2.namedWindow('Original image',cv2.WINDOW_NORMAL)
cv2.imshow('Original image',image)

#lower=np.array([54,0,0])
#upper=np.array([120,255,255])    #area=100000
lower=np.array([0,65,0])
upper=np.array([179,255,255])
mask = cv2.inRange(imgHSV,lower,upper)
gray[mask==0] = 255


_,trehsold = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)


cv2.namedWindow('Thresh image',cv2.WINDOW_NORMAL)
cv2.imshow('Thresh image',trehsold)
#cv2.imwrite("Treshold.jpg", trehsold)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]


idx = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    #print(area)
    if(area >5000):
       idx +=1
       x,y,w,h = cv2.boundingRect(cnt)
       cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
   # hull = cv2.convexHull(cnt)


       print("Object:", idx+1, "x1:", x, "x2:", x+ w, "y1:", y , "y2:", y +h)
       roi=image[y:y+h,x:x+w]
       #cv2.imwrite("crop_"+str(idx) + ".jpg", roi)
       #cv2.imshow("crop_image", roi)
cv2.imshow("detected_image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()























