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
        
        img = cv2.imread(f,1)
        
        
        BLUR = 3
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 200
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (1.0,1.0,1.0) # In BGR format
        
        
        #== Processing =======================================================================
        
        #-- Read image -----------------------------------------------------------------------
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        
        
        imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower=np.array([0,85,0])
        upper=np.array([179,255,255])
        m = cv2.inRange(imgHSV,lower,upper)
        imageGray[m==0] = 255
            
                
        _,trehsold = cv2.threshold(imageGray, 250, 255, cv2.THRESH_BINARY)
        
        
        #-- Edge detection -------------------------------------------------------------------
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        
        cv2.imshow('img_edge', edges) 
        
        cv2.imshow('img_thresh', trehsold)                                   # Display
        #cv2.waitKey(0)
        
        #-- Find contours in edges, sort by area ---------------------------------------------
        contour_info = []
        contours,_ = cv2.findContours(trehsold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Previously, for a previous version of cv2, this line was: 
        #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Thanks to notes from commenters, I've updated the code but left this note
        
        
        
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]
        
        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))
        
        #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
        
        #-- Blend masked img into MASK_COLOR background --------------------------------------
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
        img         = img.astype('float32') / 255.0                 #  for easy blending
        
        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
        
        cv2.imshow('img_masked', masked)                                   # Display
        cv2.waitKey(0)
        
        cv2.imwrite('ElmaZemin/elma_kesik'+str(a) +"_"+".jpg", masked)           # Save
        
        
       
        
           
                
cv2.waitKey(0)  
cv2.destroyAllWindows()




                    
    
        
