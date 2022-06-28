import cv2
import numpy as np


# Read image
small = cv2.imread("jatniel-tunon-ZVRuAo9viVs-unsplash.jpg")
image = cv2.resize(small, (0,0), fx=1.7, fy=1.7)

# Select ROI
r = cv2.selectROI("select the area", image)

# Crop image with the coordinates
cropped_image = image[int(r[1]):int(r[1]+r[3]), 
                      int(r[0]):int(r[0]+r[2])]

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
image_sharp = cv2.filter2D(src=cropped_image, ddepth=-1, kernel=kernel)

# Display cropped image
cv2.imshow("Sharped image", image_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
