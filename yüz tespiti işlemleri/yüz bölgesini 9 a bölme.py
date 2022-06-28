# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:04:46 2022

@author: Gitek_Micro
"""

import cv2
import sys
import time
import random



# Get user supplied values
imagePath = "wan.jpg"
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
print("Görüntü Boyutları : {} x {} ".format(image.shape[1],image.shape[0]))
#image2 = cv2.resize(image, (1280,720))
#cv2.imshow("Resize image",image2)
#cv2.imwrite("emrullah_hdx250.jpg", image2)
#cv2.namedWindow("Original image",cv2.WINDOW_NORMAL)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_time = time.time()
# Detect faces in the image
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=9,minSize=(50, 50)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)
end_time=time.time()
print("Gecen Süre : ", end_time - start_time) 
print("Görüntü Boyutları : {} x {} ".format(image.shape[1],image.shape[0]))
print("Found {0} faces!".format(len(faces)))
i = 0
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    print("Yüzün x1 :{} ve y1 : {}".format(x, y))
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    kalanw = w%16
    kalanh = h%16
    w = (16 - kalanw) + w
    h = (16 - kalanh) + h
    roi=image[y:y+h,x:x+w]
    cv2.imshow("rect"+str(x),roi)
    print("Yüzün Boyutları : {} x {} ".format(roi.shape[1],roi.shape[0]))
    i+=1
    for l in range(1,5,1):
        for k in range(1, 5, 1):#satır ilerlemesi
            rect1 = image[y+ int((l-1)*h/4):y+int(l*h/4),x+int((k-1)*w/4):x+int(k*w/4)]
            cv2.imshow("rect1"+str(l)+str(k),rect1)
            cv2.waitKey(100)
   
        #print(kalanw,"\n")
        #print("Yeni genislik",h,"\n")

               
#cv2.imwrite("sifreli864x864.jpg", roi)
#cv2.imwrite("sifreli864x864.jpeg", image)
#cv2.namedWindow("Faces found",cv2.WINDOW_NORMAL)
#cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
