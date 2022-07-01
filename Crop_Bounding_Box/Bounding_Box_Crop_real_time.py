# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:13:22 2022

@author: Gitek_Micro
"""
from pypylon import pylon

import cv2
import numpy as np
from datetime import datetime
import time
# Pypylon get camera by serial number
serial_number = '24092594'
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
# camera.AcquisitionFrameRateAbs.SetValue(True)
# camera.AcquisitionFrameRateEnable.SetValue = True
# camera.AcquisitionFrameRateAbs.SetValue(30.0)
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
idx =0 
counter = 1
start_t = time.time()
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # cv2.imwrite('images\Orijinal_image_'+str(counter)+'.jpg', img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # cv2.imwrite('gray_image.jpg', img)
      
        counter +=1
        start1 = datetime.now()


        #image = cv2.imread("gige_Blob_Cetvel122.tiff",0)
        #orijinal görüntü
        #cv2.imshow('Original image',image)
        #cv2.waitKey(0)
        
        #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,trehsold = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        cv2.imwrite('images\Orijinal_image_'+str(counter)+'.jpg', trehsold)
        
        cv2.imshow('Thresh image',trehsold)
        
        contours, hierarchy = cv2.findContours(trehsold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        #cnt = contours[0]

        for cnt in contours:
              area = cv2.contourArea(cnt)
              print(area)
              if(area >500):
                  idx += 1
                  x,y,w,h = cv2.boundingRect(cnt)
                  # roi=image[y:y+h,x:x+w]
                  #cv2.imwrite(str(idx) + '.jpg', roi)
        #           cv2.rectangle(image,(x,y),(x+w,y+h),(0,250,0),1)
        # #         cv2.imwrite('blob_fistik.jpg', image)
        
        #           cv2.imshow('img bounding',roi)
                  # cv2.waitKey(0)
        print("Blob Sayısı {}".format(idx))
        # cv2.destroyAllWindows()
        
        
        
        # areas = [cv2.contourArea(c) for c in contours]
        
        #print(len(areas))
        
        # max_index = np.argmax(areas)
        #print(max_index)
        #print(areas[max_index])
        # cnt=contours[max_index]
        # x,y,w,h = cv2.boundingRect(cnt)
        # roi=image[y:y+h,x:x+w]
        # cv2.imwrite(str(idx) + '.jpg', roi)
        # end_t = time.time()
        # end1 = datetime.now()
        
        #cv2.rectangle(image,(x,y),(x+w,y+h),(2,0,250),2)
        
        #cv2.imshow('img bounding',image)
        # time_taken = end1 - start1
        
        # time_taken = int(time_taken.total_seconds() * 1000) # milliseconds
        
        # print('Time: ',time_taken) 
        
        # print("emrullah time :" ,time)
        if cv2.waitKey(1) & 0xFF == ord("q") or counter == 501:
            end_t = time.time()
            time = (end_t - start_t)
            print("emrullah time :" ,time)
            grabResult.Release()
            break
 
    
# Releasing the resource    
camera.StopGrabbing()
# camera.Release()
camera.Close()
cv2.destroyAllWindows()
     
      