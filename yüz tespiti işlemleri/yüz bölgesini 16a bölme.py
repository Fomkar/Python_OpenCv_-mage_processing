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
    rect1 = image[y:y+int(h/4),x:x+int(w/4)]
    cv2.imshow("rect1",rect1)
    cv2.waitKey(100)
    
    rect2 = image[y:y+int(h/4),x+int(w/4):x+int(2*w/4)]
    cv2.imshow("rect2",rect2)
    cv2.waitKey(100)
    
    rect3 = image[y:y+int(h/4),x+int(2*w/4):x+int(3*w/4)]
    cv2.imshow("rect3",rect3)
    cv2.waitKey(100)
    
    rect4 = image[y:y+int(h/4),x+int(3*w/4):x+int(4*w/4)]
    cv2.imshow("rect4",rect4)
    cv2.waitKey(100)
    
    rect5 = image[y+int(h/4):y+int(2*h/4),x:x+int(w/4)]
    cv2.imshow("rect5",rect5)
    cv2.waitKey(100)
    
    rect6 = image[y+int(h/4):y+int(2*h/4),x+int(w/4):x+int(2*w/4)]
    cv2.imshow("rect6",rect6)
    cv2.waitKey(100)
    
    rect7 = image[y+int(h/4):y+int(2*h/4),x+int(2*w/4):x+int(3*w/4)]
    cv2.imshow("rect7",rect7)
    cv2.waitKey(100) 
    
    rect8 = image[y+int(h/4):y+int(2*h/4),x+int(3*w/4):x+int(4*w/4)]
    cv2.imshow("rect8",rect8)
    cv2.waitKey(100)
    
    rect9 = image[y+int(2*h/4):y+int(3*h/4),x:x+int(w/4)]
    cv2.imshow("rect9",rect9)
    cv2.waitKey(100)
    
    rect10 = image[y+int(2*h/4):y+int(3*h/4),x+int(w/4):x+int(2*w/4)]
    cv2.imshow("rect10",rect10)
    cv2.waitKey(100)
    
    rect11 = image[y+int(2*h/4):y+int(3*h/4),x+int(2*w/4):x+int(3*w/4)]
    cv2.imshow("rect11",rect11)
    cv2.waitKey(100)
    
    rect12 = image[y+int(2*h/4):y+int(3*h/4),x+int(3*w/4):x+int(4*w/4)]
    cv2.imshow("rect12",rect12)
    cv2.waitKey(100)
    
    rect13 = image[y+int(3*h/4):y+int(4*h/4),x:x+int(w/4)]
    cv2.imshow("rect13",rect13)
    cv2.waitKey(100)
    
    rect14 = image[y+int(3*h/4):y+int(4*h/4),x+int(w/4):x+int(2*w/4)]
    cv2.imshow("rect14",rect14)
    cv2.waitKey(100)
    
    rect15 = image[y+int(3*h/4):y+int(4*h/4),x+int(2*w/4):x+int(3*w/4)]
    cv2.imshow("rect15",rect15)
    cv2.waitKey(100)
     
    rect16 = image[y+int(3*h/4):y+int(4*h/4),x+int(3*w/4):x+int(4*w/4)]
    cv2.imshow("rect16",rect16)
    cv2.waitKey(100)
    #print(kalanw,"\n")
    #print("Yeni genislik",h,"\n")
    for i in range(0, roi.shape[0]):
       for j in range(0, roi.shape[1]):
               roi[i][j] = random.randint(0, 255)
               
#cv2.imwrite("sifreli864x864.jpg", roi)
#cv2.imwrite("sifreli864x864.jpeg", image)
#cv2.namedWindow("Faces found",cv2.WINDOW_NORMAL)
#cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
