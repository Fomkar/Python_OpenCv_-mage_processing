# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:03:37 2021

@author: Gitek_Micro
"""
import cv2 # opencv kütüphanesi
import numpy as np #matris ve dizi kütüphanesi
from datetime import datetime #zaman kütüphanesi
import time

start_t = time.time()

start1 = datetime.now()


image = cv2.imread("Basler_acA1440-73gc__40038474__20220325_150952932_0448.bmp",0)
#orijinal görüntü
# cv2.imshow('Original image',image)
# cv2.waitKey(0)

# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,trehsold = cv2.threshold(image,60, 255, cv2.THRESH_BINARY)
# cv2.imshow('Thresh image',trehsold)
# cv2.waitKey(0)  
# cv2.destroyAllWindows()
#%%

contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

#cnt = contours[0]
idx =0 
for cnt in contours:
      area = cv2.contourArea(cnt)
      print(area)
      if(area >500):
          idx += 1
          x,y,w,h = cv2.boundingRect(cnt)
          roi=image[y:y+h,x:x+w]
          #cv2.imwrite(str(idx) + '.jpg', roi)
          cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,250),1)
#         cv2.imwrite('blob_fistik.jpg', image)

          # cv2.imshow('img bounding'+str(idx),roi)
          # cv2.waitKey(0)
print("Blob Sayısı {}".format(idx))
# cv2.destroyAllWindows()

# cv2.imshow('Original image',image)

areas = [cv2.contourArea(c) for c in contours]

#print(len(areas))

max_index = np.argmax(areas)
#print(max_index)
#print(areas[max_index])
cnt=contours[max_index]
x,y,w,h = cv2.boundingRect(cnt)
roi=image[y:y+h,x:x+w]
# cv2.imwrite(str(idx) + '.jpg', roi)
end_t = time.time()
end1 = datetime.now()

#cv2.rectangle(image,(x,y),(x+w,y+h),(2,0,250),2)

#cv2.imshow('img bounding',image)
time_taken = end1 - start1
time = end_t - start_t
time_taken = int(time_taken.total_seconds() * 1000) # milliseconds

print('Time: ',time_taken) 

print("Miliseconds time :" ,time)
cv2.imwrite("detected_image.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()