# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:51:13 2022

@author: Gitek_Micro
"""
# Yüz Tespiti
# -*- coding: utf-8 -*-


import cv2
from pypylon import pylon
import numpy as np

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
camera.AcquisitionFrameRateAbs.SetValue(10.0)
# camera.AcquisitionFrameRateAbs.SetValue = 5.0
# camera.Width.SetValue(720)
# camera.Height.SetValue = 540.0
# camera.Width.SetValue = 720.0
# camera.Width = 720
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


images = np.zeros((1000, 1080, 1440, 3), dtype=int)
# images = np.zeros((100, 540, 720, 3), dtype=int)
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier(
    "haarcascade_mcs_mouth.xml")


org = (30,50)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.5

counter = 0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # Get grabbed image
        # print("görüntü almaya başladı")




        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray image",gray_image)
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 7)
    
        if(len(faces) == 0):
            cv2.putText(img, "No Found Face", org, fontFace, 
                    fontScale, (0,0,255), 2)
        else:
            for x, y, w, h in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), 
                          (255,0,0), 2)
                roi_gray = gray_image[y:y+h, x:x+w]
            
           
                cv2.putText(img, "Face Detected :)", org, fontFace,
                            fontScale, (0,255,0), 2, cv2.LINE_AA)
              
                           
    
        cv2.imshow("Mask Detection",img)
        # print("görüntü almaya başladı")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("by")
        break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
# camera.Release()
camera.Close()
cv2.destroyAllWindows()
