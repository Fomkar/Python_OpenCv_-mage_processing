import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

currentDir = os.getcwd()
print(currentDir)
currentDir = "C:/Users\Gitek_Micro\Desktop\Karisik_Hiyar"   
os.chdir(currentDir)
files = os.listdir()
for f in files:
    if f.endswith(".bmp"):
        print(f)
        
        # The input image.
        image = cv2.imread(f, 0)
        image = cv2.resize(image,(1024,1024))
        
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Define thresholds
        #Can define thresholdStep. See documentation. 
        params.minThreshold = 60
        params.maxThreshold = 255
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 150
        params.maxArea = 800000
        
        # Filter by Color (black=0)
        params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
        params.blobColor = 0
        
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.5
        params.maxCircularity = 10
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.maxConvexity = 1
        
        # Filter by InertiaRatio
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 5
        
        # Distance Between Blobs
        params.minDistBetweenBlobs = 8
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        
        print("Number of blobs detected are : ", len(keypoints))
        
        img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for i in range(len(keypoints)):
            x = keypoints[i].pt[0] #i is the index of the blob you want to get the position
            y = keypoints[i].pt[1]
            # x1 = keypoints[i].pt[]
            blob_size = keypoints[0].size
            print("x : {}, y: {}".format(x, y))
        
        cv2.circle(img_with_blobs, (int(x),int(y)), 3, (0,0,255),-1)
        cv2.putText(img_with_blobs, "Blob Size : {}".format(int(blob_size)), (5,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
        cv2.putText(img_with_blobs, "Number of Blob : {}".format(len(keypoints)), (5,150), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,10,200),2)
        cv2.line(img_with_blobs, (int(x) - int(blob_size/2),int(y)), (int(x)+ int(blob_size/2),int(y)), (80,150,80), 2)
        
        cv2.line(img_with_blobs, (int(x),int(y) - int(blob_size/2)), (int(x),int(y) + int(blob_size/2)), (80,150,80), 2)
        
        h = int(blob_size/2)
        w =h
        h1 = int(y) - int(blob_size/2)-2
        h2 = int(y) + int(blob_size/2)+2
        w1 = int(x) - int(blob_size/2)-2
        w2 = int(x) + int(blob_size/2)+2
        crop = image[h1:h2,w1:w2]
        cv2.imshow("crop_image", crop)
        #plt.imshow(img_with_blobs)
        cv2.imshow("Keypoints", img_with_blobs)
        cv2.waitKey(500)
        if cv2.waitKey(20)==ord("k"):
           cv2.destroyAllWindows()
           break        





#%%
currentDir = os.getcwd()
print(currentDir)
currentDir = "C:/Users\Gitek_Micro\Desktop\Karisik_Hiyar"   
os.chdir(currentDir)
files = os.listdir()
for f in files:
    if f.endswith(".bmp"):
          print(f)
          img = cv2.imread(f)  
          print("{} \n".format(img.shape))
          cv2.imshow("resim1",img)
          cv2.waitKey(500)
          if cv2.waitKey(20)==27:
             cv2.destroyAllWindows()
             break
cv2.waitKey(0)
cv2.destroyAllWindows()