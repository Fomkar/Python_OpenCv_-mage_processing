# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:27:49 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np

# Read in the image in grayscale
img = cv2.imread('30-legos.jpg', cv2.IMREAD_GRAYSCALE)

# Determine which openCV version were using
if cv2.__version__.startswith('2.'):
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()

# Detect the blobs in the image
keypoints = detector.detect(img)
print(len(keypoints))

# Draw detected keypoints as green circles
imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display found keypoints
cv2.namedWindow("Keypoints",cv2.WINDOW_NORMAL)
cv2.imshow("Keypoints", imgKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()