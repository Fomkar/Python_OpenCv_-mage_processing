print('Importing SiSo Wrapper')
import os
import cv2 
import numpy as np

# IMPORT additional modules
import sys
import time
from datetime import datetime
import os

#tensorflow
import tensorflow as tf
from time import sleep
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

try:
    sys.path.append(os.path.join(os.environ['SISODIR5'],
                                 'SDKWrapper/PythonWrapper/python36/lib/'))
    sys.path.append(os.path.join(os.environ['SISODIR5'],
                                 'SDKWrapper/PythonWrapper/python36/bin/'))

    import SiSoPyInterface as s
except ImportError:
    raise ImportError('SiSo module not loaded successfully')
    

# for "s.getArrayFrom", to handle grabbed image as NumPy array
print('Importing NumPy', end='')
import numpy as np
print('Version', np.__version__)

#%% Kamera Ayarları

def getNrOfBoards():
	nrOfBoards = 0
	(err, buffer, buflen) = s.Fg_getSystemInformation(None, s.INFO_NR_OF_BOARDS, s.PROP_ID_VALUE, 0)
	if (err == s.FG_OK):
		nrOfBoards = int(buffer)
        
	return nrOfBoards

def selectBoardDialog():
	maxNrOfboards = 10
	nrOfBoardsFound = 0
	nrOfBoardsPresent = getNrOfBoards()
	maxBoardIndex = -1
	minBoardIndex = None

	if (nrOfBoardsPresent <= 0):
		print("No Boards found!")
		return -1
	
	print('Found', nrOfBoardsPresent, 'Board(s)')
	
	for i in range(0, maxNrOfboards):
		skipIndex = False
		boardType = s.Fg_getBoardType(i);
		if boardType == s.PN_MICROENABLE4AS1CL:
			boardName = "MicroEnable IV AS1-CL"
		elif boardType == s.PN_MICROENABLE4AD1CL:
			boardName = "MicroEnable IV AD1-CL"
		elif boardType == s.PN_MICROENABLE4VD1CL:
			boardName = "MicroEnable IV VD1-CL"
		elif boardType == s.PN_MICROENABLE4AD4CL:
			boardName = "MicroEnable IV AD4-CL"
		elif boardType == s.PN_MICROENABLE4VD4CL:
			boardName = "MicroEnable IV VD4-CL"
		elif boardType == s.PN_MICROENABLE4AQ4GE:
			boardName = "MicroEnable IV AQ4-GE"
		elif boardType == s.PN_MICROENABLE4VQ4GE:
			boardName = "MicroEnable IV VQ4-GE"
		elif boardType == s.PN_MICROENABLE5AQ8CXP6B:
			boardName = "MicroEnable V AQ8-CXP"
		elif boardType == s.PN_MICROENABLE5VQ8CXP6B:
			boardName = "MicroEnable V VQ8-CXP"
		elif boardType == s.PN_MICROENABLE5VD8CL:
			boardName = "MicroEnable 5 VD8-CL"
		elif boardType == s.PN_MICROENABLE5AD8CL:
			boardName = "MicroEnable 5 AD8-CL"
		elif boardType == s.PN_MICROENABLE5AQ8CXP6D:
			boardName = "MicroEnable 5 AQ8-CXP6D"
		elif boardType == s.PN_MICROENABLE5VQ8CXP6D:
			boardName = "MicroEnable 5 VQ8-CXP6D"
		elif boardType == s.PN_MICROENABLE5AD8CLHSF2:
			boardName = "MicroEnable 5 AD8-CLHS-F2"
		elif boardType == s.PN_MICROENABLE5_LIGHTBRIDGE_ACL:
			boardName = "MicroEnable 5 LB-ACL"
		elif boardType == s.PN_MICROENABLE5_LIGHTBRIDGE_VCL:
			boardName = "MicroEnable 5 LB-VCL"
		elif boardType == s.PN_MICROENABLE5_MARATHON_ACL:
			boardName = "MicroEnable 5 MA-ACL"
		elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_SP:
			boardName = "MicroEnable 5 MA-ACX-SP"
		elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_DP:
			boardName = "MicroEnable 5 MA-ACX-DP"
		elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_QP:
			boardName = "MicroEnable 5 MA-ACX-QP"
		elif boardType == s.PN_MICROENABLE5_MARATHON_AF2_DP:
			boardName = "MicroEnable 5 MA-AF2-DP"
		elif boardType == s.PN_MICROENABLE5_MARATHON_VCL:
			boardName = "MicroEnable 5 MA-VCL"
		elif boardType == s.PN_MICROENABLE5_MARATHON_VCX_QP:
			boardName = "MicroEnable 5 MA-VCX-QP"
		elif boardType == s.PN_MICROENABLE5_MARATHON_VF2_DP:
			boardName = "MicroEnable 5 MA-VF2-DP"
		else:
			boardName = "Unknown / Unsupported Board"
			skipIndex = True
		
		if not skipIndex:
			sys.stdout.write("Board ID " + str(i) + ": " + boardName + " 0x" + format(boardType, '02X') + "\n")
			nrOfBoardsFound = nrOfBoardsFound + 1
			maxBoardIndex = i
			if minBoardIndex == None: minBoardIndex = i
			
		if nrOfBoardsFound >= nrOfBoardsPresent:
			break

		if nrOfBoardsFound < 0:
			break
	
	if nrOfBoardsFound <= 0:
		print("No Boards found!")
		return -1
	
	inStr = "=====================================\n\nPlease choose a board[{0}-{1}]: ".format(minBoardIndex, maxBoardIndex)
	userInput = input(inStr)

	while (not userInput.isdigit()) or (int(userInput) > maxBoardIndex):
		inStr = "Invalid selection, retry[{0}-{1}]: ".format(minBoardIndex, maxBoardIndex)
		userInput = input(inStr)

	return int(userInput)

board_id = selectBoardDialog()
print(board_id)

if board_id < 0:
    print("not selected board !!")
    exit(1)


# definition of resolution
width = 1024
height = 1024
samplePerPixel = 1
bytePerSample = 1
isSlave = False
useCameraSimulator = True
camPort = s.PORT_A

# number of buffers for acquisition
nbBuffers = 4
totalBufferSize = width * height * samplePerPixel * bytePerSample * nbBuffers

# number of image to acquire
nrOfPicturesToGrab = 100
frameRate = 10

# initialize hub
hub_path = "median_blop.hap"
fg = s.Fg_InitEx(hub_path, board_id, 0);

# error handling
err = s.Fg_getLastErrorNumber(fg)
mes = s.Fg_getErrorDescription(err)

if err < 0:
	print("Error", err, ":", mes)
	sys.exit()
else:
	print("ok")

# allocating memory
memHandle = s.Fg_AllocMemEx(fg, totalBufferSize, nbBuffers)


# Set Applet Parameters
err = s.Fg_setParameterWithInt(fg, s.FG_WIDTH, width, camPort)
if (err < 0):
	print("Fg_setParameter(FG_WIDTH) failed: ", s.Fg_getLastErrorDescription(fg))
	s.Fg_FreeMemEx(fg, memHandle)
	s.Fg_FreeGrabber(fg)
	exit(err)

err = s.Fg_setParameterWithInt(fg, s.FG_HEIGHT, height, camPort)
if (err < 0):
	print("Fg_setParameter(FG_HEIGHT) failed: ", s.Fg_getLastErrorDescription(fg))
	s.Fg_FreeMemEx(fg, memHandle)
	s.Fg_FreeGrabber(fg)
	exit(err)

# Read back settings
(err, oWidth) = s.Fg_getParameterWithInt(fg, s.FG_WIDTH, camPort)
if (err == 0):
	print('Width =', oWidth)
(err, oHeight) = s.Fg_getParameterWithInt(fg, s.FG_HEIGHT, camPort)
if (err == 0):
	print('Height =', oHeight)
(err, oString) = s.Fg_getParameterWithString(fg, s.FG_HAP_FILE, camPort)
if (err == 0):
	print('Hap File =', oString)


# create a display window
dispId0 = s.CreateDisplay(8 * bytePerSample * samplePerPixel, width, height)
s.SetBufferWidth(dispId0, width, height)



cur_pic_nr = 0
last_pic_nr = 0
img = "will point to last grabbed image"
nImg = "will point to Numpy image/matrix"

win_name_img = "Source Image (SiSo Runtime)"
win_name_res = "Result Image (openCV)"

# start acquisition - görüntü almaya başlıoruz.
err = s.Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, s.ACQ_STANDARD, memHandle)
if (err != 0):
     print('Fg_AcquireEx() failed:', s.Fg_getLastErrorDescription(fg))
     s.Fg_FreeMemEx(fg, memHandle)
     s.CloseDisplay(dispId0)
     s.Fg_FreeGrabber(fg)
     exit(err)


# definition of resolution
width = 1024
height = 1024
samplePerPixel = 1
bytePerSample = 1
isSlave = False
useCameraSimulator = True
camPort = s.PORT_A

# number of buffers for acquisition
nbBuffers = 4
totalBufferSize = width * height * samplePerPixel * bytePerSample * nbBuffers

# number of image to acquire
nrOfPicturesToGrab = 1000
frameRate = 10

# initialize hub
hub_path = "median_blop.hap"
fg = s.Fg_InitEx(hub_path, board_id, 0);

# error handling
err = s.Fg_getLastErrorNumber(fg)
mes = s.Fg_getErrorDescription(err)

if err < 0:
	print("Error", err, ":", mes)
	sys.exit()
else:
	print("ok")

# allocating memory
memHandle = s.Fg_AllocMemEx(fg, totalBufferSize, nbBuffers)


# Set Applet Parameters
err = s.Fg_setParameterWithInt(fg, s.FG_WIDTH, width, camPort)
if (err < 0):
	print("Fg_setParameter(FG_WIDTH) failed: ", s.Fg_getLastErrorDescription(fg))
	s.Fg_FreeMemEx(fg, memHandle)
	s.Fg_FreeGrabber(fg)
	exit(err)

err = s.Fg_setParameterWithInt(fg, s.FG_HEIGHT, height, camPort)
if (err < 0):
	print("Fg_setParameter(FG_HEIGHT) failed: ", s.Fg_getLastErrorDescription(fg))
	s.Fg_FreeMemEx(fg, memHandle)
	s.Fg_FreeGrabber(fg)
	exit(err)

# Read back settings
(err, oWidth) = s.Fg_getParameterWithInt(fg, s.FG_WIDTH, camPort)
if (err == 0):
	print('Width =', oWidth)
(err, oHeight) = s.Fg_getParameterWithInt(fg, s.FG_HEIGHT, camPort)
if (err == 0):
	print('Height =', oHeight)
(err, oString) = s.Fg_getParameterWithString(fg, s.FG_HAP_FILE, camPort)
if (err == 0):
	print('Hap File =', oString)


# create a display window
dispId0 = s.CreateDisplay(8 * bytePerSample * samplePerPixel, width, height)
s.SetBufferWidth(dispId0, width, height)



cur_pic_nr = 0
last_pic_nr = 0
img = "will point to last grabbed image"
nImg = "will point to Numpy image/matrix"

win_name_img = "Source Image (SiSo Runtime)"
win_name_res = "Result Image (openCV)"

print("Acquisition started")
#total_time = []
one_image_time = []

# start acquisition - görüntü almaya başlıoruz.
err = s.Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, s.ACQ_STANDARD, memHandle)
if (err != 0):
     print('Fg_AcquireEx() failed:', s.Fg_getLastErrorDescription(fg))
     s.Fg_FreeMemEx(fg, memHandle)
     s.CloseDisplay(dispId0)
     s.Fg_FreeGrabber(fg)
     exit(err)

while cur_pic_nr < nrOfPicturesToGrab:
        cur_pic_nr = s.Fg_getLastPicNumberBlockingEx(fg, last_pic_nr + 1, camPort, 5, memHandle)
        print(cur_pic_nr)
        if (cur_pic_nr < 0):
            print("Fg_getLastPicNumberBlockingEx(", (last_pic_nr + 1), ") failed: ", (s.Fg_getLastErrorDescription(fg)))
            # s.g_stopAcquire(fg, camPort)
            # s.g_FreeMemEx(fg, memHandle)
            # s.loseDisplay(dispId0)
            s.g_FreeGrabber(fg)
            exit(cur_pic_nr)
    
        last_pic_nr = cur_pic_nr
    
        # get image pointer
        img = s.Fg_getImagePtrEx(fg, last_pic_nr, camPort, memHandle)

        # handle this as Numpy array (using same memory, NO copy)
        frame = s.getArrayFrom(img, width, height)
        
        img_width=frame.shape[1]
        img_height=frame.shape[0]
        frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)  # Görüntüyü 4 boyutlu tensöre çevirme işlemi.

        labels = ["mask","no mask"]


        colors=["0,255,255","255,0,0","255,255,0","0,255,0"]
        colors=[np.array(color.split(",")).astype("int") for color in colors]
        colors=np.array(colors) # Tek bir array de tuttuk.
        colors=np.tile(colors,(18,1)) # Büyütme işlemi yapıyoruz.
    
    
        cfg="yolov4_tiny.cfg"
        weights="yolov4_tiny_detector_last.weights"
        model=cv2.dnn.readNetFromDarknet(cfg,weights)
    
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    
        layers=model.getLayerNames()
        output_layer=[layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()] # Modelde ki çıktı katmanlarını aldık.

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
            max_class_id = max_id[0]
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
        # out.write(frame)
      
    
    
        # if cv2.waitKey(1) & 0xff == ord("q"):
        #     camera.StopGrabbing()
        #     # camera.Release()
        #     camera.Close()
        #     # out.release()
        #     break



cv2.destroyAllWindows()