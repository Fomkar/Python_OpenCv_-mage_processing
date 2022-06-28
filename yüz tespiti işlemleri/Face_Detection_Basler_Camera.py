# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:31:14 2022

@author: Nezih Önal
"""



# importing libraries
from pypylon import pylon
import cv2
import numpy as np
from pypylon import genicam


# class CameraEventPrinter(pylon.CameraEventHandler):
#     def OnCameraEvent(self, camera, userProvidedId, node):
#         print("OnCameraEvent event for device ", camera.GetDeviceInfo().GetModelName())
#         print("User provided ID: ", userProvidedId)
#         print("Event data node name: ", node.GetName())
#         value = genicam.CValuePtr(node)
#         if value.IsValid():
#             print("Event node data: ", value.ToString())
#         print()

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
    

    
# # Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
# camera.AcquisitionFrameRateAbs.SetValue(True)
# # camera.AcquisitionFrameRateEnable.SetValue = True
#camera.AcquisitionFrameRateAbs.SetValue(50.0)
# # camera.AcquisitionFrameRateAbs.SetValue = 5.0



#♣camera.Width.SetValue(1024)
# # camera.Width.SetValue = 720.0
# # camera.Width = 720;
converter = pylon.ImageFormatConverter()

# # converting to opencv bgr format
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
faceCascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")



images = np.zeros((1024, 1024, 1024, 3), dtype=int)

# images = np.zeros((100, 540, 720, 3), dtype=int)
print("Grabe başlatacak")
counter = 0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30))
		#flags = cv2.CV_HAAR_SCALE_IMAGE
        print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# Display the resulting frame
    # cv2.imshow('frame', frame)
  
        # Get grabbed image
    print(" Görünü Boyutları  x :{} y :{} ".format(frame.shape[0],frame.shape[1]))
    cv2.namedWindow("original frame", cv2.WINDOW_NORMAL)
    cv2.imshow("original frame",gray)
        
    if cv2.waitKey(1) & 0xFF == ord("k"):
            print(" \n  ======== Kamera Kapatıldı ============ \n ")
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
# camera.Release()
camera.Close()
cv2.destroyAllWindows()





