# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:12:58 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
from pypylon import pylon
import os
import time
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Pypylon get camera by serial number
# serial_number = '40038474'

height = 45;
width = 28;
x1 = 292;
y1 = 233;

    # faces[0].height = 63;
    # faces[0].width = 42;
    # faces[0].x = 211;
    # faces[0].y = 151;


frame = cv2.imread("gray.jpg",1)


cv2.imshow("gray image",frame)

# image = cv2.rectangle(frame, (292,233), (x1 + width, y1 + height), (255,255,0), 2)
image = cv2.rectangle(frame, (292,233), (x1 + width, y1 + height), (10), 12)
cv2.imshow(" image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()