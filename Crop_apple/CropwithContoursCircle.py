# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:07:38 2022

@author: Ünal
"""


import cv2 # Burada kütüphane yükleme
import numpy as np
import os


files = os.listdir() 
for f in files:
    if f.endswith(".jpg"):
        print(f)
        x = f.split("_")
        # print(x[7])
        y  = x[2].split(".")
        # print(y)
        # print(y[0])
        a = x[1] + y[0]
        print(a)
        
        image = cv2.imread(f,1)
        #image = cv2.resize(image,(640,480))
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
        

        contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
#       cnt = contours[0]

        idx =0 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            if(area >5000):
                idx += 1
                            
                circles = cv2.HoughCircles(trehsold, cv2.HOUGH_GRADIENT,4,10000,param1=100,param2=4,minRadius=4,maxRadius=7000)

                if circles is not None:
                    circles = np.round(circles[0,:]).astype("int")
                    if len(circles) == 1:
                        x,y,r = circles[0] 
                        r=r+20
                        mask = np.zeros((image.shape[0],image.shape[1],3),np.uint8)
                        cv2.circle(mask,(x,y),r,(255,255,255),-1,8,0)
                        out = image*mask
                        out2 = 255-out
                        white = 255-mask
                        cv2.imshow("detected_image", out2 + white)
                        #cv2.imwrite('ElmaContur/Elma_'+ str(idx)+ "_" + str(a) +"_"+ str(idx)+".jpg", out2+white)
                        cv2.waitKey(0)
              
                
cv2.waitKey(0)  
cv2.destroyAllWindows()




                    
    
        
