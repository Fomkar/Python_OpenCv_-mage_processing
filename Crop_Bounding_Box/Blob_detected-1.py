# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 08:40:50 2021

@author: Gitek_Micro
"""

import numpy as np
import cv2
import glob
 

blobsNotFound = []
images = glob.glob('*.tiff')
print("Fora giriyor")
for fname in images:
    print("fora girdi")
    print(fname)
    orig_img = cv2.imread(fname)
 

 
    #mask = cv2.erode(mask, None, iterations=1)
    # commented out erode call, detection more accurate without it
 
    # dilate makes the in range areas larger
    mask = cv2.dilate(orig_img, None, iterations=1)
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
    im_with_keypoints = cv2.drawKeypoints(orig_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Origin image",orig_img)
if cv2.waitKey(0) == ord("k"):
    cv2.destroyAllWindows()


#%%
import cv2
import numpy as np
  
# Load image
image = cv2.imread('Basler_raL2048-48gm__23810476__20210805_182324434_0024.tiff', 0)
image =cv2.bitwise_not(image,1)
# Set our filtering parameters
# Initialize parameter settiing using cv2.SimpleBlobDetector
detector = cv2.SimpleBlobDetector()

# Detect blobs
keypoints = detector.detect(image)
  
# Draw blobs on our image as red circles
blank = np.zeros((1, 1)) 
blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  
number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
  
# Show blobs
cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread('Basler_raL2048-48gm__23810476__20210805_182324434_0024.tiff')
edges = cv2.Canny(img,200,300,True)
cv2.imshow("Edge Detected Image", edges)  
cv2.imshow("Original Image", img)  
cv2.waitKey(0)  # waits until a key is pressed  
cv2.destroyAllWindows()  # destroys the window showing image




















