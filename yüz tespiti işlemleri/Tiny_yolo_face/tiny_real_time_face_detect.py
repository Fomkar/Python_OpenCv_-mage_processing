import cv2
import numpy as np
from pypylon import pylon
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Pypylon get camera by serial number
# serial_number = '40038474'
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
# Kamera KAyıt Alma
fourrc = cv2.VideoWriter_fourcc(*'XVID')

out= cv2.VideoWriter("mask2.avi",fourrc,10.0,(1440,1080))
frame = None

while True:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        # Access the image data
        frame = converter.Convert(grabResult)
        frame = frame.GetArray()
        # Get grabbed image

    img_width=frame.shape[1]
    img_height=frame.shape[0]
    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)  # Görüntüyü 4 boyutlu tensöre çevirme işlemi.

    labels = ["FACE"]


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
    out.write(frame)
      
    
    

    if cv2.waitKey(1) & 0xff == ord("q"):
        camera.StopGrabbing()
        # camera.Release()
        camera.Close()
        out.release()
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
    