# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:51:00 2021

@author: Gitek_Micro
"""
import statistics
import cv2
import os
import numpy as np;




# liste = [5,78,9,6,485,3,8]
# liste.sort()

# a = int(len(liste)/2)

# print("Listenin medyanı : " + str(liste[a]))



# statistics.median(liste)

# kar = [50,50,40,10,100,50,100,60,105,50,90,70,8,41,635,54]

# statistics.mode(kar) # Mod en çok tekrar eden

# statistics.quantiles(kar)
#  Folder with images
directory = 'C:/Users/Gitek_Micro/Desktop/Blob Crop'
 
for filename in os.listdir(directory):
    if filename.endswith(".tiff"): 
        image_path = os.path.join(directory, filename)
        print(filename)
        original_image = cv2.imread(filename)
        original_image = cv2.resize(original_image, (700,500))
        cv2.imshow("Original İmage",original_image)
        im = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector()
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 60;
        params.maxThreshold = 200;
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
        	detector = cv2.SimpleBlobDetector(params)
        else : 
        	detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = params.detect(im)
        
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
if cv2.waitKey(0) == ord("k"):
    cv2.destroyAllWindows()
    
#%% Deneme Alanı
# imports
import cv2
import numpy as np;

# Read image
img = cv2.imread('Basler_raL2048-48gm__23810476__20210805_182324434_0024.tiff', cv2.IMREAD_GRAYSCALE)
frame=cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("Gaussian", frame)
if cv2.waitKey(0) == ord("k"):
    cv2.destroyAllWindows()
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

keypoints = detector.detect(img)


# Detect blobs from the image.
keypoints = detector.detect(frame)
blank = np.zeros(0,0)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
blobs = cv2.drawKeypoints(img, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imshow('Blobs',blobs)
cv2.waitKey(0)

        

