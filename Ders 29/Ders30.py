# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 08:49:30 2021

@author: Ömer Karagöz
->Detected  face
"""
import cv2
import time

start_time = time.time()
#içe aktar - Read İmage
img = cv2.imread("team.jpg")
# cv2.namedWindow("original image",cv2.WINDOW_NORMAL)
# cv2.imshow("original image",img)
#gray image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("Gray image",cv2.WINDOW_NORMAL)
# cv2.imshow("Gray image", img_gray)

#sınıflandırıcı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(img_gray, minNeighbors = 10)
i=1
for(x,y,w,h) in face_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,250),3)
    cv2.putText(img, "{}.".format(i), (x+w,y+h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2 ,(0,255,255),3)
    i+=1
cv2.putText(img, "Takimda {} kisi vardir".format(len(face_rect)), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2 ,(255,255,255),2)
# cv2.namedWindow("Detected image",cv2.WINDOW_NORMAL)
# cv2.imshow("Detected image",img)
# cv2.imwrite("Barcelona Team.jpg",img)

cap = cv2.VideoCapture(0)

while True :
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame,minNeighbors = 7)
        for(x,y,w,h) in face_rect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,250),3)
            cv2.putText(frame, "Omer Karagoz", (x+w,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2 ,(0,255,255),3)
        cv2.imshow("face detected", frame)
    if cv2.waitKey(20)==ord("k"):
       break
   
    
cap.release()
print("--- %s seconds ---" % (time.time() - start_time))
cv2.destroyAllWindows()














