# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:48:30 2022

@author: Gitek_Micro
"""

import numpy as np
import cv2

import os

# Görüntüleri okuma ve gösterme
currentDir = 'D:\Zeynep_Hoca_Kodları'
print(currentDir)
os.chdir(currentDir)
files = os.listdir()
idx =0 
for i,f in enumerate(files):
    if f.endswith(".bmp"):
        # The input image.
        print(f)
        image = cv2.imread(f, 1)
      
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        imgHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower=np.array([0,51,0])
        upper=np.array([39,255,255])
        m = cv2.inRange(imgHSV,lower,upper)
        m = cv2.dilate(m, (3,3),iterations = 12)
        m = cv2.erode(m, (5,5),iterations = 9)
        m = cv2.dilate(m, (9,9),iterations = 7)
        cv2.imshow("HSV_1", m)
        cv2.waitKey(0)
        _,trehsold = cv2.threshold(m, 40, 255, cv2.THRESH_BINARY)
        
        
        # cv2.imshow("Treshold image", trehsold)
        # cv2.waitKey(0)
        contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        #cnt = contours[0]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            x,y,w,h = cv2.boundingRect(cnt)
            if(area >5500):
                idx += 1
                
                roi=image[y:y+ h,x:x+w]
                #cv2.imwrite("C:/Users/Gitek_Micro/Desktop/findik_icleri/findik_" + str(idx) + '.bmp', roi)
                #cv2.rectangle(image,(x,y),(x+w,y+h),(0,25,255),5)
                cv2.imshow('img bounding',roi)
                cv2.waitKey(0)
cv2.destroyAllWindows()