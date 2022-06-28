# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:50:28 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("elma_kesik2022-03-1112-55-49_3.jpg",1)

cv2.imshow("original_image",image)
cv2.waitKey(0)

image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("grey_image", image_grey)
cv2.waitKey(0)

image_thresh = cv2.threshold(image_grey,140,255,cv2.THRESH_BINARY)[1]
cv2.imshow("thresh_image", image_thresh)
cv2.waitKey(0)

image_binary = cv2.bitwise_not(image_thresh)
cv2.imshow("binary_image", image_binary)
cv2.waitKey(0)
"""
# Generate variables
x1,y1,w,h = cv2.boundingRect(image_binary)
x2 = x1+w
y2 = y1+h

# Draw bounding rectangle
start = (x1, y1)
end = (x2, y2)
colour = (255, 0, 0)
thickness = 1
rectangle_img = cv2.rectangle(image, start, end, colour, thickness)
roi=image[y1:y2,x1:x2]
cv2.imshow("crop_image", roi)
cv2.waitKey(0)
print("x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
plt.imshow(rectangle_img, cmap="gray")
"""

contours = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
colour = (255, 0, 0)
thickness = 1
i = 0
for cntr in contours:
    x1,y1,w,h = cv2.boundingRect(cntr)
    x2 = x1+w
    y2 = y1+h
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
    print("Object:", i+1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
    i += 1
cv2.imshow("detected_image",image)
cv2.waitKey(0)

plt.imshow(image)


cv2.destroyAllWindows()