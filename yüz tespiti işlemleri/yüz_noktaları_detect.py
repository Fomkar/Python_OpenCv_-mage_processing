# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:28:56 2022

@author: Gitek_Micro
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing =mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
cv2.namedWindow("Facemesh",cv2.WINDOW_NORMAL)
drawing_spec = mp_drawing.DrawingSpec(thickness =2,circle_radius =1)
dosya_yolu ="C:/Users/Gitek_Micro/Desktop/FaceDetect-master/productionface_2.mp4"
cap = cv2.VideoCapture(dosya_yolu)
with mp_face_mesh.FaceMesh(
         min_detection_confidence =0.5,
         min_tracking_confidence =0.5) as face_mesh:
     while cap.isOpened():
         success,image = cap.read()
         #image = cv2.resize(image, (700,640))
         if not success:
             print('Video yükelmedi.')
             continue
         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
         
         
         results = face_mesh.process(image)
         
         #Yüz görüntüsünün üzerine noktaları çizelim
         
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         if results.multi_face_landmarks:
             for face_landmarks in results.multi_face_landmarks:
                 mp_drawing.draw_landmarks(
                     image=image, landmark_list=face_landmarks,
                     connections=(mp_face_mesh.FACEMESH_CONTOURS),
                     landmark_drawing_spec=drawing_spec,
                     connection_drawing_spec=drawing_spec)
                 mp_drawing.draw_landmarks(
                     image=image, landmark_list=face_landmarks,
                     connections=(mp_face_mesh.FACEMESH_FACE_OVAL),
                     landmark_drawing_spec=drawing_spec,
                     connection_drawing_spec=mp_drawing_styles
                     .get_default_face_mesh_tesselation_style())
             cv2.imshow("Facemesh", image)
         if cv2.waitKey(5) & 0xFF == 27:
             break
cap.release()
cv2.destroyAllWindows()