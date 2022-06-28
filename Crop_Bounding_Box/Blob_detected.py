# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 08:40:50 2021

@author: Gitek_Micro
"""

import numpy as np
import cv2
import glob
 

blobsNotFound = []
images = glob.glob('Blob Crop\\*.tiff')
 
for fname in images:
    print(fname)
    orig_img = cv2.imread(fname)
 
    # undistort and crop
    dst = cv2.undistort(orig_img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    crop_frame = dst[y:y+h, x:x+w]
    # Blur image to remove noise
    frame=cv2.GaussianBlur(crop_frame, (3, 3), 0)
 
    # Switch image from BGR colorspace to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of purple color in HSV
    purpleMin = (115,50,10)
    purpleMax = (160, 255, 255)
    
    # Sets pixels to white if in purple range, else will be set to black
    mask = cv2.inRange(hsv, purpleMin, purpleMax)
        
    # Bitwise-AND of mask and purple only image - only used for display
    res = cv2.bitwise_and(frame, frame, mask= mask)
 
    #mask = cv2.erode(mask, None, iterations=1)
    # commented out erode call, detection more accurate without it
 
    # dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=1)
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
     
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
     
    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5
     
    detector = cv2.SimpleBlobDetector_create(params)
 
    # Detect blobs.
    reversemask=255-mask
    keypoints = detector.detect(reversemask)
    



























