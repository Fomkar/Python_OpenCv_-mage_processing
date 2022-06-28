# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:38:48 2022

@author: Gitek_Micro
"""

import cv2
import numpy as np
from pypylon import pylon
import os
import time
from glob import glob

path_images = glob("C:/Users/Gitek_Micro/Desktop/yuz_veriseti/cnv_3/*.jpeg")

path_images.sort(key= lambda x:int(x[52:-5]))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Pypylon get camera by serial number
# serial_number = '40038474'

for i in range(1133): 
    frame = cv2.imread(path_images[i])
    img_width=frame.shape[1]
    img_height=frame.shape[0]
    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)  # Görüntüyü 4 boyutlu tensöre çevirme işlemi.

    labels = ["face","face"]


    colors=["255,0,0","255,0,0"]
    colors=[np.array(color.split(",")).astype("int") for color in colors]
    colors=np.array(colors) # Tek bir array de tuttuk.
    colors=np.tile(colors,(18,1)) # Büyütme işlemi yapıyoruz.
    
    
    cfg="yolov4-tiny-pretrained.cfg"
    weights="yolov4-tiny-pretrained.weights"
    model=cv2.dnn.readNetFromDarknet(cfg,weights)
    
    # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    
    layers=model.getLayerNames()
    output_layer=[layers[layer-1] for layer in model.getUnconnectedOutLayers()] # Modelde ki çıktı katmanlarını aldık.

    model.setInput(frame_blob)

    detection_layers=model.forward(output_layer)

    #----------- Non Maximum Supression Operation-1 ----------
    ids_list=[]
    boxes_list=[]
    confidence_list=[]
    #------------ End Of Opertation 1 -------------

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores=object_detection[5:]
            predicted_id=np.argmax(scores)
            confidence=scores[predicted_id]
            if confidence > 0.30:
                label=labels[predicted_id]
                bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
                
                start_x=int(box_center_x-(box_width/2))
                start_y =int(box_center_y - (box_height / 2))

                # ----------- Non Maximum Supression Operation-2 ----------
                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
                # ------------ End Of Opertation 2 -------------

    # ----------- Non Maximum Supression Operation-3 ----------
    max_ids=cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4)
    
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
        print("start x:{}  y : {}  width : {}  high : {}".format(start_x,start_y,box_width,box_height))
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_class_id]
        # ------------ End Of Opertation 3 -------------

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = "{}: {:.2f}%".format(label, confidence * 100)
        print("Predicted_object: ", label)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 3)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_ITALIC, 0.6, box_color, 2)

    t, _ = model.getPerfProfile()
    text = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Detection",frame)   
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break


cv2.destroyAllWindows()


# Fps hesaplama:
    # start = time.time()
    # net.setInput(blob)
    # detections = net.forward(net.getUnconnectedOutLayersNames()
    # end = time.time()
    # ms_per_image = (end - start) * 1000 / 100
    # print("Time per inference: %f ms" % (ms_per_image))
    # print("FPS: ", 1000.0 / ms_per_image)
    