# -*- coding: utf-8 -*-
"""
@author: Mustafa Ünlü
@instagram: mmustafaunluu
@youtube: Kendi Çapında Mühendis
"""

import cv2
import numpy as np
from random import randint as rnd
from pypylon import pylon








def nothing(x):
    pass

cv2.namedWindow("frame")
cv2.createTrackbar("H1", "frame", 0, 359, nothing)
cv2.createTrackbar("H2", "frame", 0, 359, nothing)
cv2.createTrackbar("S1", "frame", 0, 255, nothing)
cv2.createTrackbar("S2", "frame", 0, 255, nothing)
cv2.createTrackbar("V1", "frame", 0, 255, nothing)
cv2.createTrackbar("V2", "frame", 0, 255, nothing)


kernel = np.ones((5,5), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX


# Pypylon get camera by serial number
serial_number = '40038474'
info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()
    

    
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
camera.AcquisitionFrameRateAbs.SetValue(True)
# camera.AcquisitionFrameRateEnable.SetValue = True
camera.AcquisitionFrameRateAbs.SetValue(30.0)
# camera.AcquisitionFrameRateAbs.SetValue = 5.0
# camera.Width.SetValue(720)
# camera.Height.SetValue = 540.0
# camera.Width.SetValue = 720.0
# camera.Width = 720
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

paint = np.ones((900,720,3), np.uint8)*255
paint = cv2.flip(paint,1)
images = np.zeros((1000, 1080, 1440, 3), dtype=int)

# images = np.zeros((100, 540, 720, 3), dtype=int)

counter = 0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # Get grabbed image
        
        if counter < 1000:
            images[counter] = img

    
  
  
    img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    H1 = int(cv2.getTrackbarPos("H1","frame") / 2)
    H2 = int(cv2.getTrackbarPos("H2","frame") / 2)
    S1 = cv2.getTrackbarPos("S1","frame")
    S2 = cv2.getTrackbarPos("S2","frame")
    V1 = cv2.getTrackbarPos("V1","frame")
    V2 = cv2.getTrackbarPos("V2","frame")
    
    
    lower = np.array([H1,S1,V1])
    upper = np.array([H2,S2,V2])
    
    mask = cv2.inRange(hsv,lower,upper)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(img,img,mask=mask)
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 50000 or area < 200:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        print(x,y,w,h)
        
        color = (rnd(0,256), rnd(0,256), rnd(0,256))

        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        
        (x2,y2), radius = cv2.minEnclosingCircle(cnt)
        center2 = (int(x2),int(y2))
        cv2.circle(img ,center2, int(radius), (255,0,0),2)
        cv2.circle(img, center, 5, (0,255,0),-1)
        cv2.circle(img, (x,y), 5, (0,0,255),-1)
        cv2.drawContours(img, contours, i, color, 4)
        # text = str((w,h))
        # cv2.putText(img, text, (x,y), font, 1, color, 2)
        cv2.circle(paint, center2, 10, color,-1)
    
   # cv2.imshow("frame",frame)
    cv2.imshow("res",res)
    cv2.imshow("img",img)
    cv2.imshow("Paint",paint)
    
    key = cv2.waitKey(5)
    
   
    
    if cv2.waitKey(5) == ord("q"):
        break
    elif cv2.waitKey(5) == ord("e"):
       paint[:] = 255     
        
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

camera.Close()
cv2.destroyAllWindows()
