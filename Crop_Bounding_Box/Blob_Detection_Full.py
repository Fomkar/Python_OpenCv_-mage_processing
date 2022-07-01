# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:08:49 2021

@author: Gitek_Micro
"""



#Countours Bulma

import cv2 
import numpy as np 
from PIL import Image
import os
from glob import glob
images = []

def get_images():
    currentDir = os.getcwd()
    print(currentDir)
    currentDir = "C:/Users\Gitek_Micro\Desktop\Fındık Veri Seti"   
    os.chdir(currentDir)
    files = os.listdir()
    
    for i in files:
        if i.endswith(".bmp"):
           images.append(i)
    return images


# Let's load a simple image with 3 black squares 

get_images()
a = 0
for i in images:
    image = cv2.imread(i,1) 
    # Find Canny edges 
    edged = cv2.Canny(image, 30, 200) 
    cv2.waitKey(0) 
       
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.imshow('Original', image) 
    #cv2.imshow('Canny Edges After Contouring', edged)  
       
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
    #cv2.putText(image, "Countrous Size :{}".format(contours.size), (10,50) , cv2.FONT_HERSHEY_DUPLEX,2, (0,0,255),1)
    
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    print( M )
    if M['m00']== 0:
        M['m00'] = 1
     
    if cv2.waitKey(0) == ord("k"):
        cv2.destroyAllWindows() 
        break
    
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        perimeter = cv2.arcLength(cnt,True)
        print(perimeter)
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # crop_image = image.crop(100,100,500,500)
        # cv2.imshow("Crop_image", crop_image)
        # face = image[cy-180:cy+150, cx-150:cx+130]
        crop = image[y:y+h,x:x+w]
        cv2.imshow("crop_image", crop)
        #cv2.imwrite("C:/Users\Gitek_Micro\Desktop\crop\Resim_{}.bmp".format(a), crop)
        a +=1;
        # cv2.imshow("Face", face)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(255,0,0),2)
        cv2.imshow('Contours', image) 
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image,center,radius,(0,0,255),2)
        cv2.imshow('Circle', image)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()       