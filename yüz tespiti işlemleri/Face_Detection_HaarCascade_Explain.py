# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 11:54:45 2022

@author: Nezih Önal
"""

import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread("Nezih.bmp")
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow('img',image)
cv2.waitKey(0)
#image = cv2.resize(image, (640,480))
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.5,9,minSize = (50,50))
for (x,y,w,h) in faces:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,2,2),5)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow('img',image)
cv2.imwrite("face Deteceted Nezih_Önal.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

